# import torch
# from modeling.models.gaitset import GaitSet
#
# # 实例化模型并加载预训练权重
# model = GaitSet()
# model.build_network(model_cfg={
#     'in_channels': [1, 32, 64, 128],
#     'SeparateFCs': {
#         'in_channels': 128,
#         'out_channels': 256,
#         'parts_num': 62
#     },
#     'bin_num': [16, 8, 4, 2, 1]
# })
# model.load_state_dict(torch.load('D:\Projects\gaitOpenGait\output\GaitSet-40000.pt'))
# model.eval()
#
# # 构造 dummy input
# dummy_input = torch.randn(1, 1, 30, 64, 44)  # N, C, S, H, W
#
# # 输出文件名
# output_onnx = "gaitset_model.onnx"
#
# # 导出 ONNX
# torch.onnx.export(
#     model,
#     (dummy_input, ),  # 注意这里需要加逗号，表示元组
#     output_onnx,
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output'],
#     dynamic_axes={
#         'input': {0: 'batch_size', 2: 'sequence_length'},
#         'output': {0: 'batch_size'}
#     }
# )
#
# print(f"Model exported to {output_onnx}")

# import torch
# from modeling.models.gaitset import GaitSet
#
# # Monkey patch: 绕过 get_msg_mgr
# import utils.msg_manager as msg_manager
# msg_manager.get_msg_mgr = lambda: None
#
# # 构造 cfgs 并实例化模型
# cfgs = {
#     'in_channels': [1, 32, 64, 128],
#     'SeparateFCs': {
#         'in_channels': 128,
#         'out_channels': 256,
#         'parts_num': 62
#     },
#     'bin_num': [16, 8, 4, 2, 1]
# }
#
# model = GaitSet(cfgs=cfgs, training=False)
# model.build_network(model_cfg=cfgs)
# model.load_state_dict(torch.load('D:\Projects\gaitOpenGait\output\GaitSet-40000.pt'))
# model.eval()
#
# # 构造 dummy input
# dummy_input = torch.randn(1, 1, 30, 64, 44)  # N, C, S, H, W
#
# # 输出文件名
# output_onnx = "gaitset_model.onnx"
#
# # 导出 ONNX
# torch.onnx.export(
#     model,
#     (dummy_input,),
#     output_onnx,
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output'],
#     dynamic_axes={
#         'input': {0: 'batch_size', 2: 'sequence_length'},
#         'output': {0: 'batch_size'}
#     }
# )
#
# print(f"Model exported to {output_onnx}")

# import torch
#
# # 在导出脚本顶部添加以下代码
# import sys
# sys.path.append('.')
#
# from modeling.models.gaitset import GaitSet
#
# # 创建一个空的 msg_mgr 替代品
# class DummyMsgManager:
#     def __getattr__(self, name):
#         return lambda *args, **kwargs: None
#
# # Monkey patch 掉整个 get_msg_mgr 模块
# import utils.msg_manager as msg_manager
# msg_manager.get_msg_mgr = lambda: DummyMsgManager()
#
#
# # 构造 cfgs 并实例化模型
# cfgs = {
#     'in_channels': [1, 32, 64, 128],
#     'SeparateFCs': {
#         'in_channels': 128,
#         'out_channels': 256,
#         'parts_num': 62
#     },
#     'bin_num': [16, 8, 4, 2, 1]
# }
#
# model = GaitSet(cfgs=cfgs, training=False)
# model.build_network(model_cfg=cfgs)
# model.load_state_dict(torch.load('D:\Projects\gaitOpenGait\output\GaitSet-40000.pt'))
# model.eval()
#
# # 构造 dummy input
# dummy_input = torch.randn(1, 1, 30, 64, 44)  # N, C, S, H, W
#
# # 输出文件名
# output_onnx = "gaitset_model.onnx"
#
# # 导出 ONNX
# torch.onnx.export(
#     model,
#     (dummy_input,),
#     output_onnx,
#     export_params=True,
#     opset_version=11,
#     do_constant_folding=True,
#     input_names=['input'],
#     output_names=['output'],
#     dynamic_axes={
#         'input': {0: 'batch_size', 2: 'sequence_length'},
#         'output': {0: 'batch_size'}
#     }
# )
#
# print(f"Model exported to {output_onnx}")

import torch
from gaitset_for_inference import GaitSetForInference

# 实例化模型
model = GaitSetForInference()

# 加载预训练权重（注意：你需要把 state_dict 映射到新模型）
state_dict = torch.load('D:\Projects\gaitOpenGait\output\GaitSet-40000.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# 构造 dummy input
dummy_input = torch.randn(1, 1, 30, 64, 44)  # N, C, S, H, W

# 输出文件名
output_onnx = "gaitset_model.onnx"

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    output_onnx,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'sequence_length'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported to {output_onnx}")

