# Experiment Data Location 

## Log

root: 

> /media/xue/Data Storage/CSI/Train/Log

下分6个子文件夹，对应参数 **rootType**，第一次需动态新建：

```
AmplitudeWithout_PhaseWith 
AmplitudeWithOut_PhaseWithout 
AmplitudeWith_PhaseWith 
AmplitudeWith_PhaseWithout 
OnlyAmplitude
OnlyPhase 
```

每一个**rootType**文件夹下分3个文件夹，对应参数which，第一次需动态新建：

```
fixed
open
semi
```

每一个which文件夹下面对应若干个子文件夹，命名为0,1,2....有几次训练就有几个文件夹，每次动态新建：

```
0
1
2
...
```

每一个子文件夹下又有两个子文件夹：

```
'train'
'val'
```

## Pb

pbRoot:

> ```
> '/media/xue/Data Storage/CSI/Train/Model'
> ```

和log类似的逻辑和文件层级关系：

下分6个子文件夹，对应参数 **rootType**，第一次需动态新建：

```
AmplitudeWithout_PhaseWith 
AmplitudeWithOut_PhaseWithout 
AmplitudeWith_PhaseWith 
AmplitudeWith_PhaseWithout 
OnlyAmplitude
OnlyPhase 
```

每一个**rootType**文件夹下分3个文件夹，对应参数which，第一次需动态新建：

```
fixed
open
semi
```

每一个which文件夹下面对应若干个子文件夹，命名为0,1,2....有几次训练就有几个文件夹，每次动态新建：

```
0
1
2
...
```

## Confusion matrix

matrixRoot:

> ```
> '/media/xue/Data Storage/CSI/Train/ConfusionMatrix'
> ```

和log类似的逻辑和文件层级关系：

下分6个子文件夹，对应参数 **rootType**，第一次需动态新建：

```
AmplitudeWithout_PhaseWith 
AmplitudeWithOut_PhaseWithout 
AmplitudeWith_PhaseWith 
AmplitudeWith_PhaseWithout 
OnlyAmplitude
OnlyPhase 
```

每一个**rootType**文件夹下分3个文件夹，对应参数which，第一次需动态新建：

```
fixed
open
semi
```

每一个which文件夹下面对应若干个子文件夹，命名为0,1,2....有几次训练就有几个文件夹，每次动态新建：

```
0
1
2
...
```

每一个子文件夹下有三个文件(训练中生成)：

```
trainPredictionLabel.txt
trainReallyLabel.txt
confusionMatrix.png

```

