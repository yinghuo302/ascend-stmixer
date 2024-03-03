# 基于昇腾部署的STMixer动作检测器

[STMixer](https://github.com/MCG-NJU/STMixer) is A One-Stage Sparse Action Detector

1. `config.py`: 一些全局配置
2. `dataprocess.py`: 数据预处理（颜色转换，缩放，shape处理）和数据后处理（画框）
3. `engine.py`: 昇腾运行模型相关代码，对外提供inference接口进行一次推理
4. `main.py`: 主程序代码
5. `videowriter.py`: 将输出视频写文件或者rtsp推流
6. `data`: 测试视频和转换后的昇腾om模型
