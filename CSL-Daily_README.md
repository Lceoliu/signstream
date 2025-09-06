## CSL-Daily Data 格式说明

文件夹结构:

```
- CSL_Daily
    - frames_512x512
        - ${name}.npy
        - ...
    - sentence-crop
        - ${name}.mp4
        - ...
    - annotations.json
    - split_files.json
```

### 数据格式

1. ${name}.npy 的 shape 为 [T_frames, 134, 3], 其中134代表1（视频WH）+133（COCO格式身体2d坐标点），3代表（x, y, score），score为该点的置信度。

2. ${name}.mp4 的编码格式为 H.264，分辨率为 512x512，帧率为 30fps。

3. annotations.json中存储了视频和对应的文本翻译和gloss信息，格式如下：

```json
{
    "${name}": {
        "text": "翻译的手语文本",
        "gloss": "翻译 手语 文本",
        "num_frames": T,
        "signer": i,
    },
    ...
}
```

4. split_files.json中存储了训练、验证和测试集的划分信息，格式如下：

```json
{
    "train": [
        "${name1}",
        "${name2}",
        ...
    ],
    "val": [
        "${name3}",
        "${name4}",
        ...
    ],
    "test": [
        "${name5}",
        "${name6}",
        ...
    ]
}
```

### 辅助说明

1. COCO数据全身关键点格式(https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png)：共133个关键点，其中：

    1-17号为躯干主要关键点，18-23为足部关键点（手语不需要），24-91为面部关键点，92-112为左手关键点，113-133为右手关键点

    - 各个部位的中心点的index：
        躯干主要关键点： 1
        面部关键点： 32
        左手关键点： 92
        右手关键点： 113

    - 可用于归一化长度的一些关键点的index：
        左眼眼睑：66，69 （即这两个点的距离为眼睑长度）
        右眼眼睑：60，63
        双眼眼距：2，3
        头宽：4，5
    
        注意：必须选择某种长度（比如双眼眼睑长度的平均值作为单位长度，对所有2d keypoint进行归一化）

2. 数据集本身将不同视频的pose npy文件分开储存了，如果需要快速读取，可以先写一个脚本将独立的npy文件合并为一个较大的numpy内存映射文件(memmap)，并创建一个索引字典来获取对应视频的pose。

3. **注意**：split_files.json中存储的name并不一定都能在annotations.json或对应的npy文件中找到对应的条目，在进行dataload的时候忽略无法对应的条目。