<a name="AQNal"></a>
## 概述
智能服务预警系统主要目的是为我们现有的客户组件提供分析预警服务，并在不改变现有组件功能基础上，智能服务预警系统能够独立提供日志的收集，分析以及预测等功能，并根据纳入算法模型，向用户提供智能化的信息预警服务。
<a name="fYGiD"></a>
## 现状与目的

**现状**<br />首先，当前并没有针对我们已经容器化的组件的预警系统，在前后端调试方面，我们为了调试方面，安装了portainer作为日志查询中是否出现故障的排查工具。但是，该工具只是容器的管理软件，并不能作为我们糖料罐、五金仓、AIOT等组件的监控系统。

其次，容器化的组件在轻量化部署等方面有很大的优势，尤其适用于在支持烟草无外网环境，和数据科学部无基础运行环境等方面，可以提前感知和解决他们的问题，从而减少现场排查的时间。但是，这也产生了一个问题，就是大量的容器组件，很难一一开发他们的预警系统，所以，我们需要一个统一的，并且支持我们现有环境的预警系统。
<a name="inyvo"></a>
##### 
**目的**

该系统主要实现如下三大目的：

- 完善我们现有组件缺乏监控系统的短板
- 开发出一个新型系统，可以统一支持我们所有的组件服务
- 该系统能够运行一些智能化的检测手段，减轻开发人员运维的压力

<a name="OaQc8"></a>
## 技术与方法

**设计方法**

- **轻量化和容器化**

     为了提高系统的灵活性和可扩展性，AI服务预警系统采用轻量化和容器化的架构。它利用容器技术，如Docker，将预警功能封装为独立的容器，使其可以在不同的环境中部署和运行。这样的架构可以降低资源消耗，提高系统的弹性和可移植性。

- **Python脚本实现AI预警和大数据视图服务**

      使用Python编写脚本，结合AI算法和大数据分析技术，实现AI预警和大数据视图服务。Python具有丰富的数据科学和机器学习库，如TensorFlow和Scikit-learn，可以支持复杂的预警模型和数据分析任务。

- **统一的镜像服务和日志报警**

使用容器技术，如Docker，实现统一的镜像服务和日志报警。使用Docker镜像可以将业务组件和AI预警脚本封装为独立的容器，便于部署和管理。通过集中管理容器日志并设置报警机制，可以及时发现和解决潜在问题。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/26815489/1685603362842-fe55a6c7-da4f-4ae6-a6b0-2d4e6b64f27d.png#averageHue=%23dee9d8&clientId=u5e786c31-663a-4&from=paste&height=436&id=iS1da&originHeight=872&originWidth=1870&originalType=binary&ratio=2&rotation=0&showTitle=false&size=116164&status=done&style=none&taskId=ubb52ef02-3d5a-4617-b9a1-11b55f6cc64&title=&width=935)<br />**技术栈**

该项目主要使用下列技术实现，实现的功能包括数据收集、数据处理、数据可视化

| 技术栈 | 说明 |
| --- | --- |
| flask | Flask是一个用于构建Web应用程序的Python框架，适合开发小型到中型规模的Web应用。 |
| mysql | MySQL是一种常见的关系型数据库管理系统，用于数据的存储和管理。 |
| plotly | 强大的web可视化库 |
| scikit-learn | 机器学习库 |


**设计思路**

在智能服务预警系统中，可以对时间、CPU占用率、内存占用率、磁盘占用率以及日志中是否出现某些keyword进行判断分析，以此来决定是否报警。<br />优势举例说明如下:<br />1.传统的报警模型，在某项阈值到达90%报警。但是此时报警已经很晚，所以我们需要根据历史增长趋势，使用机器学习算法，算出历史趋势来，提前预警。

2.传统的报警模型，往往依据某一个，或者两三个项目阈值机械判断当前是否应该报警。但是这种方式，需要根据不定时修改，整体的设置比较繁琐，而且无法兼顾所有的情况，比如无法结合日志与影响报警因素（内存，资源占用率）综合判断。

**预测流程**

预警系统周期性获得系统数据，将数据喂入算法库，根据算法预测结果来生成是否预警的信息，并根据用户提前设置的指令，选择是否将预警信息发送给用户。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/26815489/1686822946258-9e6432c2-a3a6-48a5-b20c-2dc20590e2f0.png#averageHue=%23faf9f9&clientId=uf743a60c-d1f1-4&from=paste&height=543&id=ua7d90fd5&originHeight=1086&originWidth=1194&originalType=binary&ratio=2&rotation=0&showTitle=false&size=86387&status=done&style=none&taskId=u25d0742b-8efc-47d1-a38d-7e3aa01c472&title=&width=597)


采用flask框架作为web服务，初始化一个定时任务，并同时启动一个web。基础代码如下：
```yaml
import schedule
import threading
import time
from flask import Flask

def job():
    print("定时任务执行中...")

def start_web_app():
    app = Flask(__name__)

    @app.route('/')
    def hello():
        return 'Hello, World!'

    app.run()

if __name__ == '__main__':
    # 初始化定时任务
    schedule.every(5).seconds.do(job)

    # 创建定时任务的线程
    schedule_thread = threading.Thread(target=lambda: schedule_loop())

    # 启动定时任务线程
    schedule_thread.start()

    # 启动Web服务
    start_web_app()

def schedule_loop():
    while True:
        schedule.run_pending()
        time.sleep(1)
```

**图形界面**

界面A:<br />界面A负责图形展示等功能，主要目的是让用户了解当前的系统状态，以及选择是否发送预警信息。<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/26815489/1686819973468-a034d61e-bc22-4666-998c-ce8bcd305524.png#averageHue=%23fefefd&clientId=ufcd9d60f-6f7f-4&from=paste&height=747&id=u94600af0&originHeight=1494&originWidth=3008&originalType=binary&ratio=2&rotation=0&showTitle=false&size=337636&status=done&style=none&taskId=u83d59804-fc6e-40d6-9075-95f904878c8&title=&width=1504)

界面B:<br />界面B负责读取需要训练数据，并根据内置算法，生成界面A中的预测模型。<br />![截屏2023-06-15 17.06.31.png](https://cdn.nlark.com/yuque/0/2023/png/26815489/1686820021155-866f3b36-677a-4daa-8802-99fff58d0627.png#averageHue=%23fefefe&clientId=ufcd9d60f-6f7f-4&from=paste&height=712&id=u1ffee160&originHeight=1424&originWidth=3016&originalType=binary&ratio=2&rotation=0&showTitle=false&size=278352&status=done&style=none&taskId=u14960332-b597-48ba-9a3b-52b3cf3c74f&title=&width=1508)

<a name="hzW3b"></a>


