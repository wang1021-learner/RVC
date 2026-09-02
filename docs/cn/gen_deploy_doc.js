const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, VerticalAlign, PageNumber, LevelFormat, TabStopType } = require("docx");
const fs = require("fs");
const path = require("path");

const FONT = "微软雅黑";
const BLUE = "1F4E79";
const BLUE2 = "2E75B6";
const ROW_ALT = "F2F7FB";
const LABEL_BG = "E7EEF6";
const WHITE = "FFFFFF";
const LINE = "8FAADC";
const BLACK = "222222";
const MUTED = "666666";

const PAGE_W = 11906;
const MARGIN = 1020;
const TW = PAGE_W - MARGIN * 2; // 9866

const thin = { style: BorderStyle.SINGLE, size: 8, color: LINE };
const borders = { top: thin, bottom: thin, left: thin, right: thin };
const headerBorder = { style: BorderStyle.SINGLE, size: 8, color: BLUE };
const headerBorders = { top: headerBorder, bottom: headerBorder, left: headerBorder, right: headerBorder };

function run(text, o = {}) {
  return new TextRun({
    text: text == null ? "" : String(text),
    font: FONT,
    size: o.size || 21,
    bold: !!o.bold,
    color: o.color || BLACK,
    italics: !!o.italics,
  });
}

function para(text, o = {}) {
  const content = Array.isArray(text)
    ? text
    : [run(text, { size: o.size, bold: o.bold, color: o.color, italics: o.italics })];
  return new Paragraph({
    alignment: o.align || AlignmentType.LEFT,
    spacing: { before: o.before ?? 40, after: o.after ?? 40, line: o.line || 276 },
    children: content,
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    spacing: { before: level === 1 ? 280 : 200, after: 120 },
    border: level === 1
      ? { bottom: { style: BorderStyle.SINGLE, size: 12, color: BLUE, space: 4 } }
      : undefined,
    children: [run(text, { size: level === 1 ? 28 : 24, bold: true, color: BLUE })],
  });
}

function asParas(children, o = {}) {
  if (Array.isArray(children)) {
    return children.map((c) => (c instanceof Paragraph ? c : para(c, o)));
  }
  if (children instanceof Paragraph) return [children];
  return [para(children, o)];
}

function cell(width, children, o = {}) {
  const paras = asParas(children, o);
  return new TableCell({
    borders: o.header ? headerBorders : borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: o.fill || WHITE, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 80, right: 80 },
    verticalAlign: o.valign || VerticalAlign.CENTER,
    columnSpan: o.span || 1,
    children: paras,
  });
}

function hcell(width, text, o = {}) {
  return cell(width, para(text, {
    bold: true, color: WHITE, size: o.size || 20,
    align: AlignmentType.CENTER, before: 20, after: 20,
  }), { fill: BLUE, header: true, valign: VerticalAlign.CENTER });
}

function lcell(width, text) {
  return cell(width, para(text, { bold: true, size: 20, before: 20, after: 20 }), {
    fill: LABEL_BG,
  });
}

function table(colWidths, rows) {
  const sum = colWidths.reduce((a, b) => a + b, 0);
  if (sum !== TW) throw new Error("col width " + sum + " != " + TW);
  return new Table({
    width: { size: TW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows,
  });
}

function kvTable(pairs) {
  const c1 = 1800, c2 = TW - 1800;
  return table([c1, c2], pairs.map(([k, v], i) => new TableRow({
    children: [
      lcell(c1, k),
      cell(c2, Array.isArray(v) ? v : para(v, { size: 20 }), {
        fill: i % 2 ? ROW_ALT : WHITE,
      }),
    ],
  })));
}

function gridTable(widths, header, body) {
  const rows = [
    new TableRow({ children: header.map((h, i) => hcell(widths[i], h)), tableHeader: true }),
    ...body.map((row, ri) => new TableRow({
      children: row.map((c, i) => cell(
        widths[i],
        Array.isArray(c) ? c : para(c, {
          size: 20,
          align: i === row.length - 1 && String(c).trim() === "□" ? AlignmentType.CENTER : AlignmentType.LEFT,
          before: 20, after: 20,
        }),
        { fill: ri % 2 ? ROW_ALT : WHITE },
      )),
    })),
  ];
  return table(widths, rows);
}

function bullet(text, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { before: 40, after: 40, line: 276 },
    children: [run(text, { size: 21 })],
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: BLUE },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: BLUE },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "b1", levels: [{ level: 0, format: LevelFormat.BULLET, text: "·", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
      { reference: "b2", levels: [{ level: 0, format: LevelFormat.BULLET, text: "·", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
      { reference: "b3", levels: [{ level: 0, format: LevelFormat.BULLET, text: "·", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 220 } } } }] },
      { reference: "n1", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 280 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_W, height: 16838 },
        margin: { top: 900, right: MARGIN, bottom: 900, left: MARGIN },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: BLUE, space: 6 } },
          spacing: { after: 80 },
          children: [
            run("RVC 实时变声", { size: 18, bold: true, color: BLUE }),
            run("    部署运维方案与现场信息收集", { size: 18, color: MUTED }),
          ],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 6, color: LINE, space: 8 } },
          spacing: { before: 80 },
          tabStops: [{ type: TabStopType.RIGHT, position: TW }],
          children: [
            run("DH-UVTC-D09  V1.0  内部资料 · 注意保存", { size: 16, color: MUTED }),
            run("\t", { size: 16 }),
            run("第 ", { size: 16, color: MUTED }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: MUTED }),
            run(" 页", { size: 16, color: MUTED }),
          ],
        })],
      }),
    },
    children: [
      para("部署运维方案与现场信息收集", {
        size: 36, bold: true, color: BLUE, align: AlignmentType.CENTER, before: 80, after: 40,
      }),
      para("RVC 实时变声 · 服务器客户端", {
        size: 24, color: BLUE2, align: AlignmentType.CENTER, before: 0, after: 80,
      }),
      para("文档编号：DH-UVTC-D09　　版本：V1.0　　密级：内部资料 · 注意保存", {
        size: 18, color: MUTED, align: AlignmentType.CENTER, before: 0, after: 160,
      }),

      table([2000, 2933, 2000, 2933], [
        new TableRow({ children: [
          lcell(2000, "编制人 / 日期"),
          cell(2933, para("研发：____________    ____年__月__日", { size: 18 })),
          lcell(2000, "审核人 / 日期"),
          cell(2933, para("技术负责人：____________    ____年__月__日", { size: 18 })),
        ]}),
        new TableRow({ children: [
          lcell(2000, "批准人 / 日期"),
          cell(2933, para("研发总监：____________    ____年__月__日", { size: 18 })),
          lcell(2000, "会签 / 日期"),
          cell(2933, para("业务 / 法务 / 安全：____________    ____年__月__日", { size: 18 })),
        ]}),
      ]),

      para("说明：签名栏请手填姓名与日期。下文按当前已交付能力编写，未做能力不写进验收范围。", {
        size: 18, color: MUTED, italics: true, before: 80, after: 80,
      }),

      heading("0  交付范围与架构", 1),
      para("本方案对应「服务器客户端 + 远端 GPU 推理」模式：客户电脑只跑界面和声卡，变声计算在 NVIDIA Tesla T4 推理机上完成。客户机不需要独立显卡，也不随包装模型文件。"),
      para("数据路径：麦克风 → 客户机 Windows 客户端 → WebSocket（TCP 8765）→ 推理机加载角色并实时变声 → 回传音频 → 耳机/扬声器（或可选虚拟声卡）。"),
      kvTable([
        ["产品形态", "Windows 桌面客户端（PySide6），不是浏览器、不是网页坐席。"],
        ["安装包", "RVC实时变声-服务器版安装.exe（Inno Setup）；亦可解压绿色包 RVC服务器客户端.zip。"],
        ["推理服务", "server/rvc_server.py，监听 0.0.0.0:8765，由 start_rvc.sh 看门狗拉起。"],
        ["协议与端口", "ws://推理机IP:8765（明文 WebSocket）。本期不上 wss、不上 WAF、不上独立接入网关。"],
        ["并发", "默认一人一机，--max-sessions 1。第二路需第二台 T4，或评估后显式加大路数。"],
        ["本期不做", "坐席账号、配置中心下发、开机自启、远程自动升级、浏览器接入、TensorRT、TLS 证书。"],
      ]),

      heading("一、移动云侧部署", 1),
      kvTable([
        ["云资源开通", [
          para("GPU：NVIDIA Tesla T4 × 1。建议规格（按云厂商实际套餐改填）：16 vCPU / 64 GB 内存 / 系统盘 ≥ 100 GB SSD / 数据盘 ≥ 200 GB。", { size: 20 }),
          para("地域：选距客户最近的可用区，降低 RTT。网络：公网 IP 或专线/VPN；单路实时变声建议预留 5～10 Mbps。", { size: 20 }),
          para("现网规格请现场填写：vCPU ____ / 内存 ____ GB / 系统盘 ____ GB / 带宽 ____ Mbps / 地域 ________。", { size: 20 }),
        ]],
        ["网络与安全", [
          para("VPC + 子网。安全组入站：TCP 8765（推理，对客户网段或公网按策略开放）；TCP 22（运维，仅跳板/办公网）。其余端口默认拒绝。", { size: 20 }),
          para("本期：ws://，不部署 WAF、不部署 TLS 证书、不采购独立 DDoS 高防。若后续改 wss://，再补证书与 443 转发。", { size: 20 }),
        ]],
        ["服务部署", [
          para("工作目录示例：/root/songwang/rvc-infer。启动：python server/rvc_server.py --host 0.0.0.0 --port 8765。", { size: 20 }),
          para("模型：assets/weights/*.pth；索引：assets/indices/*.index。客户端只传文件名，远端必须有同名文件。", { size: 20 }),
          para("看门狗：start_rvc.sh 循环拉起进程。CUDA 上下文损坏（illegal memory access 等）进程主动退出，由看门狗重启；10 分钟内第 3 次起等待 30 秒，第 6 次起等待 120 秒。", { size: 20 }),
        ]],
        ["GPU 环境", [
          para("NVIDIA 驱动 + CUDA 11.8 + Python 3.11 + PyTorch cu118。T4 推理精度：FP16。启用 CUDA Graph 预热。", { size: 20 }),
          para("不使用 TensorRT。公共特征：HuBERT、音高 RMVPE、检索 FAISS Top-K=4。", { size: 20 }),
        ]],
        ["数据区", [
          para("模型与索引存放推理机本地盘，不随客户安装包分发。定期备份 assets/weights、assets/indices。", { size: 20 }),
          para("服务日志：rvc_server.log。客户机角色名单与地址：%AppData%\\RVC实时变声\\（覆盖安装不冲掉）。", { size: 20 }),
        ]],
        ["扩容", "当前为有状态单机推理，不是无状态网关。扩容方式：新增 T4 实例、部署同一套服务、客户机改填新地址。禁止在未评估 GPU 占用前把单机 max-sessions 盲目加大。"],
      ]),

      heading("二、现场客户端部署", 1),
      para("2.0  安装与配置", { size: 24, bold: true, color: BLUE, before: 80, after: 80 }),
      bullet("客户电脑：Windows 10/11 64 位，麦克风 + 耳机/扬声器，能访问推理机 TCP 8765。不需要本机 NVIDIA 显卡。", "b1"),
      bullet("安装：双击「RVC实时变声-服务器版安装.exe」，按向导安装到当前用户目录。也可解压绿色包，直接运行 RVC实时变声.exe。", "b1"),
      bullet("打开软件 → 右侧勾选「远程服务器」→ 地址填 ws://推理机IP:8765 → 点「连接」→ 选角色 → 点「开始变声」。", "b1"),
      bullet("音高、检索、共振峰可现场微调，保存在本机 AppData。无坐席账号，无需登录。", "b1"),
      bullet("虚拟声卡不是安装包内置项。仅当变声需要送进微信/其它软件时，再单独安装 VB-CABLE，并把客户端输出选到 CABLE Input。", "b1"),
      bullet("本期无开机自启、无远程自动升级。版本更新由实施人员替换安装包；客户已有 AppData 角色名单不会被覆盖，更新角色需删除 %AppData%\\RVC实时变声\\speakers.json 后重开。", "b1"),

      para("预置角色（安装包内名单，模型在服务器）：", { size: 21, before: 120, after: 80 }),
      gridTable(
        [3200, 3333, 3333],
        ["界面显示名", "模型文件", "索引文件"],
        [
          ["myvoice 200轮", "myvoice.pth", "myvoice.index"],
          ["shanxi", "shanxi_e200_s11800.pth", "shanxi.index"],
          ["OP2694892", "OP2694892_e50_s4900.pth", "OP2694892_added_IVF1676_…v2.index"],
          ["OP2701992", "OP2701992_e50_s4950.pth", "OP2701992_added_IVF1590_…v2.index"],
          ["OP2655302 王娟", "OP2655302_e200_s22200.pth", "OP2655302_added_IVF1887_…v2.index"],
          ["OP2671842 爼海峰", "OP2671842_e200_s20400.pth", "OP2671842_added_IVF1602_…v2.index"],
          ["OP2687374 牛志鹏", "OP2687374_e200_s22600.pth", "OP2687374_added_IVF2131_…v2.index"],
          ["OP2701067 刘智贤", "OP2701067_e200_s28800.pth", "OP2701067_added_IVF2531_…v2.index"],
        ],
      ),
      para("大模型首次切换可能需要十几秒，属正常加载，不是断网。状态栏应保持「已连接」，切勿按防火墙故障处理。", {
        size: 18, color: MUTED, before: 80, after: 80,
      }),

      heading("2.1  现场上线前检查清单", 2),
      gridTable(
        [2200, 6466, 1200],
        ["检查项", "标准", "确认"],
        [
          ["客户机系统", "Windows 10/11 64 位；已安装本方案对应版本客户端。", "□"],
          ["到推理机网络", "能访问 ws://推理机IP:8765。建议 RTT < 50 ms（同城/专线更佳）。安全组已放行 TCP 8765。", "□"],
          ["连接状态", "点「连接」后状态为「已连接 · Tesla T4 ·（角色名）」，不是「连接超时 / 安全组」误报。", "□"],
          ["耳机麦克风", "系统能识别设备；客户端输入/输出下拉能选到；佩戴耳机，避免扬声器回授。", "□"],
          ["角色加载", "能选中预置角色并加载完成（大模型允许十几秒）。换角色后可变声。", "□"],
          ["实时变声", "点「开始变声」有输出；卡顿数值可接受；无持续炸麦、无长时间静音。", "□"],
          ["虚拟声卡", "非必须。仅当要送入其它软件时：已装 VB-CABLE，输出选 CABLE Input。不需要则填「不适用」。", "□"],
          ["本机存储", "角色与地址写入 %AppData%\\RVC实时变声\\；重装软件后名单仍在。", "□"],
        ],
      ),
      para("说明：本产品不是浏览器坐席，检查清单不含 Chrome/Edge、不含「在线 WEB 系统里选虚拟声卡」。", {
        size: 18, color: MUTED, before: 80, after: 40,
      }),

      heading("三、灰度发布", 1),
      gridTable(
        [1400, 2800, 2833, 2833],
        ["阶段", "范围", "观察", "进 / 退条件"],
        [
          ["灰度 1", "1 台客户机、1 路，连现网 T4，真实对讲 1～2 小时。", "连通、换角色、时延/卡顿、炸麦、掉线、CUDA 是否被看门狗拉起。", "连续 2 小时无断连且换模型可恢复 → 进入灰度 2。出现无法恢复的断连或必崩模型 → 回退，查 rvc_server.log。"],
          ["灰度 2", "仍按单路验收。若需第 2 路：另备 T4，或评估后调整 max-sessions，不得默认同机双路。", "同上，加长时间稳定与指定角色音质（王娟 / 爼海峰 / 牛志鹏 / 刘智贤）。", "连续 5 个工作日达标 → 验收全量。不达标则维持灰度 1 或回退。"],
        ],
      ),

      heading("四、现场信息收集（实施填写）", 1),
      para("以下由实施人员在现场或开通后填写，作为开通记录。"),
      kvTable([
        ["客户名称", "________________________________"],
        ["现场地址", "________________________________"],
        ["现场联系人 / 电话", "____________ / ____________________"],
        ["客户机数量 / 系统", "____ 台　　Windows 版本：____________________"],
        ["推理机公网/内网 IP", "____________________　　客户端填写：ws://________:8765"],
        ["云主机规格", "vCPU ____ / 内存 ____ GB / 磁盘 ____ GB / 带宽 ____ Mbps / 地域 ________"],
        ["安全组放行", "TCP 8765 已放行：是 / 否　　来源网段：____________________"],
        ["实际使用角色", "________________________________"],
        ["是否使用虚拟声卡", "否 / 是（软件名称：__________，输出设备：__________）"],
        ["开通日期 / 实施人", "____年__月__日　　实施人：____________"],
        ["遗留问题", "________________________________"],
      ]),

      heading("五、运维应急", 1),
      gridTable(
        [2600, 7266],
        ["现象", "处理"],
        [
          ["客户端提示连接超时，且尚未连上", "查地址是否 ws://IP:8765；本机能否访问该 IP；云安全组/防火墙是否放行 TCP 8765；推理进程是否在监听。"],
          ["已显示已连接，换角色失败或报连接超时", "属加载等待，不是防火墙。大模型允许十几秒。再点一次角色；反复失败再重新「连接」。"],
          ["变声中断、服务无响应", "查推理机进程。看门狗应自动拉起。必要时在服务器执行：pkill -f 'server/rvc_server.py'（由看门狗重启）。"],
          ["显卡 illegal memory access", "进程会主动退出交给看门狗。短时间反复崩溃应查是否某个模型必崩，查看 rvc_server.log，不要连续手动狂点加载。"],
          ["客户看不到新人名", "删除该电脑 %AppData%\\RVC实时变声\\speakers.json 后重开客户端，会从安装包拷贝最新名单。"],
          ["角色加载提示找不到文件", "核对服务器 assets/weights、assets/indices 是否有同名 .pth / .index，客户端只传文件名。"],
        ],
      ),

      para("附录  客户使用步骤（可撕给现场）", { size: 24, bold: true, color: BLUE, before: 280, after: 80 }),
      bullet("安装并打开「RVC实时变声」。", "n1"),
      bullet("勾选远程服务器，填写 ws://推理机IP:8765，点「连接」。", "n1"),
      bullet("在角色列表选择对应人员（如 OP2701067 刘智贤），等待加载完成。", "n1"),
      bullet("选择麦克风和耳机，点「开始变声」。", "n1"),
      bullet("卸载：开始菜单「卸载 RVC实时变声」，或 Windows 设置 → 应用。", "n1"),
    ],
  }],
});

const out = path.join(__dirname, "RVC实时变声-部署运维方案与现场信息收集.docx");
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length);
});
