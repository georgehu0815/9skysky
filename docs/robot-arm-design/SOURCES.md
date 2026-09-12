# 来源、网站案例与证据边界

核查日期：2026-09-12。外部页面用于物料/接口/案例调查；本项目尚未购入或测试这些机械臂。除用户拍卖页面外，优先使用厂商和项目官方材料。

## 外部硬件与开源案例

### S1 用户指定的拍卖套件

https://www.nellisauction.com/p/Robotic-Arm-Kit-6DOF-Programming-Robot-Arm-with-5-Servo-Handle/71774071

- 页面品牌字段为 LewanSoul，正文将产品描述为 LeArm。
- 标题“5 Servo”与特性段“6 个数字舵机”不一致；这是可观察到的页面冲突，不是已解析的实物关节清单。
- 用于参考金属支架、桌面夹爪、手柄/PC 教学形态。不据此指定电源、持续负载、协议、型号修订、可读反馈和附件完整性。
- 页面为已结束拍卖；历史成交/零售价不是本方案采购报价。

### S2 ROBOTIS XL330-M288-T 官方规格

https://docs.robotis.com/docs/dxl/model_reference/x_series/xl_series/xl330-m288/

- 本设计使用的官方字段：18 g、3.7–6.0 V（推荐 5.0 V）、5 V 堵转 0.52 N·m / 1.47 A、TTL 半双工、支持位置/电流相关控制和反馈。
- 官方温度范围上限 70°C。本文更低的警告/停止温度只是项目提案，不能当厂商连续运行认证。
- 官方提醒其电流测量有具体实现限制；电流到输出力矩/指尖力需校准。所有力矩预算均另标工程假设，不把堵转值作持续负载能力。
- 当前 Microduck 源码证实 XL330 系列控制器；并不替代逐颗实物型号鉴别，尤其不能默认其所有现有电机均为候选 M288-T。

### S3 ROBOTIS U2D2 官方接口

https://emanual.robotis.com/docs/en/parts/interface/u2d2/

- 支持 TTL 和 RS-485 等接口；本文选 TTL 口匹配候选 XL330。
- U2D2 不给 DYNAMIXEL 提供电源；需外部电源，并注意地参考与连接正确性。
- 不推断它自带电隔离、电源保护或能直接接任意三针 hobby servo。

### S4 Hiwonder LeArm AI 官方教程（版本对照，不是拍卖实物说明书）

https://docs.hiwonder.com/projects/LeArm_AI/en/latest/docs/1.Geting_Ready.html

- 可学习组装准备、不同舵机/控制方式和软件教程的组织方式。
- 当前 LeArm AI 文档不证明用户拍卖 LeArm 具有相同主控、舵机组合、电源或协议。本文未将其规格移植到拍卖套件。

### S5 Hugging Face LeRobot SO-101 官方案例

https://huggingface.co/docs/lerobot/so101

- 适合学习开源 arm 的装配、leader/follower 示教、舵机身份及标定流程。
- 官方页面 follower 使用六颗 STS3215；这是另一条执行器生态，不能当 XL330/Dynamixel 线序和协议兼容实例。
- 本阶段不引入 LeRobot 依赖、不将其数据集/模型成功率作为 RLX/PPO 结果。

### S6 Google DeepMind MuJoCo Menagerie

https://github.com/google-deepmind/mujoco_menagerie

值得查看的模型目录：`trs_so_arm100`（单臂形态）、`aloha`（双臂工作站）、`ufactory_lite6`（六姿态关节工业/桌面臂参照）。

- 借鉴 MJCF 资产组织、关节链、夹爪和桌面双臂布局，不等于这些模型对应本项目硬件。
- SO-ARM100 与 SO-101 不混称为同一硬件模型；本文不假定其网格、限位和惯量可无修改互换。
- 项目说明各模型子目录可能有各自许可证，复用/分发前逐目录检查并记录 commit 与资产哈希；不能只看仓库顶层许可。
- 本阶段没有下载并验证这些模型，也没有声称其演示是本项目生成的视频。

## 本地源码证据

Amazon 补充研究见 [AMAZON.md](AMAZON.md)：用户搜索页的读取限制、可核对商品身份与关键词入口分别标注；规格仍以 ALITOVE、Waveshare 和 ROBOTIS 官方资料为准，不以搜索排序或宣传标题推断兼容性。

路径相对于 `/Volumes/ExternalSSD/geoagent/microduck-lab`。读取时 upstream `microduck` HEAD 为 `bc41fb5`，`rlx` HEAD 为 `05acab3`；工作区含未提交变更，实际后续工件应额外绑定源文件哈希。

| 编号 | 文件 | 核查点 |
|---|---|---|
| L1 | `microduck/duck-control/src/model.rs:11` | `NUM_JOINTS=15`、ID、mouth 索引与电池估算边界 |
| L2 | `microduck/duck-control/src/obs.rs:1`；`microduck/duck-control/src/policy.rs:332` | 61/14 策略、mouth omission、严格加载检查 |
| L3 | `microduck/duck-control/src/bus.rs:1`；`microduck/deploy/robotd.toml:15` | XL330、同步读写、原 UART、控制时序 |
| L4 | `microduck/docs/design/architecture.md:22`；`microduck/duck-control/src/safety.rs:1`；`microduck/duck-ipc-proto/src/lib.rs:291` | 身体控制权、Safety、JSON-RPC 意图接口 |
| L5 | `duck-viewer/lib/experiments.ts:1`；`duck-viewer/lib/rlx-job.ts:215` | 八案例注册、实验特定工件和命令边界 |
| L6 | `duck-viewer/lib/evaluation.ts:3`；`duck-viewer/lib/rlx-render-evidence.ts:12` | fail-closed 技能状态、视频/源文件绑定 |
| L7 | `docs/drawing-case/brush/README.md:1` | 当前彩笔 PPO 与 BC 40/40，模拟范围和归因边界 |
| L8 | `rlx/rlx/environments/microduck.py:97` | shared adapter 拒绝非 61/14，不适合直接塞入臂关节 |
| L9 | `rlx/rlx/environments/brush.py:19` | 新任务的独立 93/15 合同先例，不意味着真机 15 动作可直接兼容 |
| L10 | `duck-viewer/app/api/rlx/artifact/route.ts:1` | 工件访问/视频 byte range 的复用面 |

## 本次不作为事实的内容

150 mm 臂长、20 g 载荷、五关节具体角范围、双臂 220 mm 间距、66/6 与 116/12 新合同、5 V 10 A 级选型、0.12/0.16 N·m 内部筛选、训练超参数、验收成功率、未来路径和新 RPC 全是**设计提案**。须依次经 CAD、模型、台架、训练和验收落实。

未提供现行总价、未证明库/SDK 与机载 OS 即插即用、未测试 Arm PPO 收敛、未生成物理 rollout 视频、未验证人机协作或移动平台负载能力。
