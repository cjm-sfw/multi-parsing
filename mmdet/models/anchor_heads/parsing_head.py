import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmdet.ops import DeformConv, roi_align
from mmdet.core import multi_apply, matrix_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule

INF = 1e8

def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).float()
    return heat * keep

def dice_loss(input, target):
    target_parsing = [(target==(i+1)).float() for i in range(14)]
    d_parsing = torch.tensor([0], device=input[0].device)
    
    
    for i in range(14):
        if input[i] == None:
            continue
        if len(target_parsing[i]) == 0:
            continue
        
        input_i = torch.sigmoid(input[i]).contiguous().view(input[i].size()[0], -1)
        target_parsing_i = target_parsing[i].contiguous().view(target_parsing[i].size()[0], -1).float()

        a = torch.sum(input_i * target_parsing_i, 1)
        b = torch.sum(input_i * input_i, 1) + 0.001
        c = torch.sum(target_parsing_i * target_parsing_i, 1) + 0.001
        d = (2 * a) / (b + c)
        
        d = (1 - d).mean()
        #print(d)
        #import pdb;pdb.set_trace()
        d_parsing = d_parsing + d/14
    
    return d_parsing

def HeatmapLoss(pred, gt):
    assert pred.size() == gt.size()
    mask = (gt>0).int()
    mask[mask == 0] = 0.1
    loss = ((pred - gt)**2) * mask
    loss = loss.mean()
    return loss


@HEADS.register_module
class ParsingHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 seg_feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128, 512)),
                 sigma=0.2,
                 num_grids=None,
                 ins_out_channels=64,
                 loss_ins=None,
                 loss_cate=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 use_dcn_in_tower=False,
                 type_dcn=None):
        super(ParsingHead, self).__init__()
        self.lambda_center = 1000
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.loss_cate = build_loss(loss_cate)
        self.ins_loss_weight = loss_ins['loss_weight']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.type_dcn = type_dcn
        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        norm_heatmap = dict(type='BN', requires_grad=True)
        self.center_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        self.kernel_parsing = nn.ModuleList()
        self.parsing_heatmap = nn.ModuleList()
        for i in range(self.stacked_convs):
            if self.use_dcn_in_tower:
                cfg_conv = dict(type=self.type_dcn)
            else:
                cfg_conv = self.conv_cfg

            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            
            
            self.kernel_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
            

            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.center_convs.append(
                ConvModule(
                    chn,
                    self.seg_feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        
        self.kernel_convert = nn.Conv2d(
            self.seg_feat_channels, 42*self.cate_out_channels, 3, padding=1)
            
        for i in range(self.cate_out_channels):
            self.kernel_parsing.append(
                nn.Sequential(
                   ConvModule(
                        42,
                        self.kernel_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None),
                    ConvModule(
                        self.kernel_out_channels,
                        self.kernel_out_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=cfg_conv,
                        norm_cfg=norm_cfg,
                        bias=norm_cfg is None),
                )
            )
            
        self.parsing_heatmap.append(
                ConvModule(
                    512,
                    256,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))

        self.parsing_heatmap.append(
            ConvModule(
                    256,
                    128,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.parsing_heatmap.append(
            ConvModule(
                    128,
                    64,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_cfg,
                    bias=norm_cfg is None))
        self.parsing_heatmap.append(
            ConvModule(
                    64,
                    1,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=cfg_conv,
                    norm_cfg=norm_heatmap,
                    bias=norm_heatmap is None))
        

    def init_weights(self):
        normal_init(self.kernel_convert, std=0.01)
        for m in self.center_convs:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_convs:
            normal_init(m.conv, std=0.01)
        for m in self.parsing_heatmap:
            normal_init(m.conv, std=0.01)
        for m in self.kernel_parsing:
            for n in m:
                normal_init(n.conv, std=0.01)
        
        #bias_cate = bias_init_with_prob(0.01)
        #normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)
        center_pred, kernel_pred = multi_apply(self.forward_single, new_feats,
                                                       list(range(len(self.seg_num_grids))),
                                                       eval=eval, upsampled_size=upsampled_size)
        return center_pred, kernel_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x
        # ins branch
        # concat coord
        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)
        
        # kernel branch
        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')

        center_feat = kernel_feat

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
            
        kernel_feat = self.kernel_convert(kernel_feat)
        
        kernel_pred = []
        
        for i,kernel_squ in enumerate(self.kernel_parsing):
            kernel_pred.append(
                kernel_squ(kernel_feat[:,i*42:(i+1)*42,:,:])
            )
        

        # center branch
        center_feat = center_feat.contiguous()
        for i, center_layer in enumerate(self.center_convs):
            center_feat = center_layer(center_feat)
        for i, heatmap_layer in enumerate(self.parsing_heatmap):
            center_feat = heatmap_layer(center_feat)
        center_pred = center_feat
        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).permute(0, 2, 3, 1)
        return center_pred, kernel_pred

    def loss(self,
             cate_preds,
             C_kernel_preds,
             ins_pred,
             gt_bbox_list,
             gt_label_list,
             gt_mask_list,
             gt_parsing_list,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        mask_feat_size = ins_pred.size()[-2:]
        
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, parsing_label_list = multi_apply(
            self.parsing_target_single,
            gt_bbox_list,
            gt_label_list,
            gt_mask_list, 
            gt_parsing_list,
            mask_feat_size=mask_feat_size)
        
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        ins_parsings = [torch.cat([parsing_labels_level_img
                                 for parsing_labels_level_img in parsing_labels_level], 0)
                      for parsing_labels_level in zip(*parsing_label_list)]
 
        C_kernel_preds = [[[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(c_kernel_preds_level, grid_orders_level)]
                          for c_kernel_preds_level in kernel_preds_level]
                        for kernel_preds_level, grid_orders_level in zip(C_kernel_preds, zip(*grid_order_list))]
    
        for imgs_idx in range(len(cate_label_list)):
            for level_img_idx in range(len(cate_label_list[imgs_idx])):
                w, h = cate_label_list[imgs_idx][level_img_idx].shape
                level_img = cate_label_list[imgs_idx][level_img_idx]
                cate_label_list[imgs_idx][level_img_idx] = cate_label_list[imgs_idx][level_img_idx].float()
                for pt in level_img.nonzero():
                    x0 = pt[0]
                    y0 = pt[1]
                    l = x0-1 if x0 > 0 else x0
                    r = x0+1 if x0 < w-1 else x0
                    t = y0-1 if y0 > 0 else y0
                    b = y0+1 if y0 < h-1 else y0
                    for sx in range(l, r+1):
                        for sy in range(t, b+1):
                            gauss_val = self.get_heat_val(2/3, sx, sy, x0, y0)
                            try:
                                cate_label_list[imgs_idx][level_img_idx][sx, sy] =  gauss_val if gauss_val > level_img[sx, sy] \
                                                                                            else level_img[sx, sy]      
                            except Exception as e:
                                print(e)
                                import pdb;pdb.set_trace()
        # generate masks
        ins_pred = ins_pred
        ins_pred_list = []
         
        
        for s_kernel_pred in C_kernel_preds:
            c_ins_pred_list = []
            for c_kernel_pred in s_kernel_pred:
                
                c_mask_pred = []
                for idx, kernel_pred in enumerate(c_kernel_pred):
                    
                    if kernel_pred.size()[-1] == 0:
                        continue
                    cur_ins_pred = ins_pred[idx, ...]
                    H, W = cur_ins_pred.shape[-2:]
                    N, I = kernel_pred.shape
                    cur_ins_pred = cur_ins_pred.unsqueeze(0)
                    kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                    cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
                    c_mask_pred.append(cur_ins_pred)
                if len(c_mask_pred) == 0:
                    c_mask_pred = None
                else:
                    c_mask_pred = torch.cat(c_mask_pred, 0)

                c_ins_pred_list.append(c_mask_pred)
            ins_pred_list.append(c_ins_pred_list)
        
        
        
        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()
        
        # dice loss
        loss_ins = torch.tensor([0], device=torch.device('cuda'))
        #import pdb;pdb.set_trace()
        for input, target in zip(ins_pred_list, ins_labels):
            if input[0] is None or target.shape[0] == 0:
                continue
            loss_ins = loss_ins + dice_loss(input, target)
        if len(loss_ins) != 0:
            loss_ins = loss_ins / len(ins_pred_list)
            loss_ins = loss_ins * self.ins_loss_weight
        else:
            loss_ins = 0
        # center

        cate_labels = [ 
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds).reshape(-1)
        
        #import pdb;pdb.set_trace()
        
        loss_center = HeatmapLoss(flatten_cate_preds, flatten_cate_labels)
        loss_center = loss_center * self.lambda_center
        
        #loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels.long(), avg_factor=num_ins + 1)
        return dict(
            loss_ins=loss_ins,
            loss_cate=loss_center)
    
    def get_heat_val(self, sigma, x, y, x0, y0):
        
        if x!=x0 or y!=y0:
            #g = 0.32
            g = torch.exp((- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)).float())
        else:
            g = 1
        return g

    def parsing_target_single(self,
                               gt_bboxes_raw,
                               gt_labels_raw,
                               gt_masks_raw,
                               gt_parsing_raw,
                               mask_feat_size):

        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        
        parsing_label_list = []
        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.seg_num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)
            
            parsing_label = []
            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                parsing_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                parsing_label_list.append(parsing_label)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices.cpu().numpy(), ...]
            gt_parsing = gt_parsing_raw[hit_indices.cpu().numpy(), ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            gt_masks_pt = torch.from_numpy(gt_masks).to(device=device)
            center_ws, center_hs = center_of_mass(gt_masks_pt)
            valid_mask_flags = gt_masks_pt.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            for seg_mask, parsing, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_parsing, gt_labels, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
                seg_mask = mmcv.imrescale(seg_mask, scale=1. / output_stride)
                seg_mask = torch.from_numpy(seg_mask).to(device=device)
                parsing = mmcv.imrescale(parsing, scale=1. / output_stride)
                parsing = torch.from_numpy(parsing).to(device=device)
                
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_parsing = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        cur_ins_parsing[:parsing.shape[0], :parsing.shape[1]] = parsing
                        
                        parsing_label.append(cur_ins_parsing)
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                parsing_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
                parsing_label = torch.stack(parsing_label, 0)
            ins_label_list.append(ins_label)
            parsing_label_list.append(parsing_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list, parsing_label_list

    def get_seg(self, cate_preds, kernel_preds, seg_pred, img_metas, cfg, rescale=None):
        num_levels = len(cate_preds)
        featmap_size = seg_pred.size()[-2:]

        result_list = []
        for img_id in range(len(img_metas)):
            cate_pred_list = [
                cate_preds[i][img_id].view(-1, self.cate_out_channels).detach() for i in range(num_levels)
            ]
            seg_pred_list = seg_pred[img_id, ...].unsqueeze(0)
            kernel_pred_list = [
                kernel_preds[i][img_id].permute(1, 2, 0).view(-1, self.kernel_out_channels).detach()
                                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            ori_shape = img_metas[img_id]['ori_shape']

            cate_pred_list = torch.cat(cate_pred_list, dim=0)
            kernel_pred_list = torch.cat(kernel_pred_list, dim=0)

            result = self.get_seg_single(cate_pred_list, seg_pred_list, kernel_pred_list,
                                         featmap_size, img_shape, ori_shape, scale_factor, cfg, rescale)
            result_list.append(result)
        return result_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds,
                       kernel_preds,
                       featmap_size,
                       img_shape,
                       ori_shape,
                       scale_factor,
                       cfg,
                       rescale=False, debug=False):

        assert len(cate_preds) == len(kernel_preds)

        # overall info.
        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > cfg.score_thr)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        import pdb;pdb.set_trace()
        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.seg_num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.seg_num_grids)
        strides[:size_trans[0]] *= self.strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_-1]:size_trans[ind_]] *= self.strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        I, N = kernel_preds.shape
        kernel_preds = kernel_preds.view(I, N, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        # mask.
        seg_masks = seg_preds > cfg.mask_thr
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.nms_pre:
            sort_inds = sort_inds[:cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                    kernel=cfg.kernel,sigma=cfg.sigma, sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= cfg.update_thr
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > cfg.max_per_img:
            sort_inds = sort_inds[:cfg.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                    size=upsampled_size_out,
                                    mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                               size=ori_shape[:2],
                               mode='bilinear').squeeze(0)
        seg_masks = seg_masks > cfg.mask_thr
        return seg_masks, cate_labels, cate_scores
