# TextDIV
诱导/非诱导的二分类模型

# 运行说明
①训练
项目内命令行运行“python train.py”，生成runs文件夹下的新文件，文件名为时间戳）
②测试运行
先更改eval.py中24行的文件名（runs下文件夹名称改为①中生成的文件夹名称，即时间戳）
运行“python eval.py --eval_train”
