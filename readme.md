MNN Batch_size 模型推理

参考 `MNN` 源码中 `demo\exec\pictureRecognition.cpp`文件，进行修改实现 `Batch_size >= 1` 进行模型推理。

模型导出时需要导出为动态 `Batch_size`
```python
model.eval()
torch.onnx.export(model, img, 
                export_file, verbose=True, opset_version=12,
                training=torch.onnx.TrainingMode.EVAL,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                            'images': {0: 'batch_size'},  # variable lenght axes
                            'output': {0: 'batch_size'}
                            }
            )
```

其他需要注意：网络结构中 `Reshape` 或者 `View`操作不能出现 `batchsize` , 需要换为 `-1`
```python
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(-1, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    _, c, num , w, h = x.shape
    #  x = x.view(batchsize, -1 , height, width) error
    x = x.view(-1, c*num , height, width) # correct

    return x
```


用一个简单地多标签分类为示例进行测试。


>参考
>https://github.com/alibaba/MNN/tree/master/demo/exec