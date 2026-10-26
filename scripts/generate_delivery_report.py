#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动车组受电弓网视频故障监测系统 - 算法交付与误报抑制评测报告生成脚本 (更新修订版)
依据: 653条全量客户现场误报视频数据基准及两种统计分析方式
修订要点:
  1. 第 1 章仅保留“监测故障类型与算法功能矩阵”，采用两列（故障监测名称、故障检测图像结果），嵌入现场检测结果抓拍图。
  2. 评测数据集统一为“653 条全量客户现场误报视频”，两种统计方式为该数据集的不同分析维度。
  3. 全文彻底去除“sheet1”、“sheet2”等表名表述，采用规范工程术语。
输出目标:
  1. /mnt/T4/jyb/gongwang_clean/reports/动车组受电弓网视频故障监测算法交付与误报抑制评测报告.docx
  2. /media/ibd/USB_DISK/10.22白/动车组受电弓网视频故障监测算法交付与误报抑制评测报告.docx
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """设置单元格内边距 (单位: twips, 20 twips = 1 pt)"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f"<w:tcMar {nsdecls('w')}>"
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f"</w:tcMar>"
    )
    tcPr.append(tcMar)


def set_cell_shading(cell, color_hex):
    """设置单元格背景颜色"""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_three_line_table(
    table, top_bottom_color="0F4C81", header_bottom_color="0F4C81"
):
    """为表格应用标准的工程三线表格式 (顶底粗线，表头下细线，无竖线)"""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f"<w:tblBorders {nsdecls('w')}>"
        f'  <w:top w:val="single" w:sz="14" w:space="0" w:color="{top_bottom_color}"/>'
        f'  <w:bottom w:val="single" w:sz="14" w:space="0" w:color="{top_bottom_color}"/>'
        f'  <w:insideH w:val="none"/>'
        f'  <w:insideV w:val="none"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f"</w:tblBorders>"
    )
    tblPr.append(borders)

    # 为表头行的单元格设置底边框
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = parse_xml(
            f"<w:tcBorders {nsdecls('w')}>"
            f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="{header_bottom_color}"/>'
            f"</w:tcBorders>"
        )
        tcPr.append(tcBorders)


def add_callout_box(doc, text, title=None):
    """在文档中添加高亮说明框"""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.5)

    # 仅左侧深蓝粗边框
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f"<w:tcBorders {nsdecls('w')}>"
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="0F4C81"/>'
        f'  <w:top w:val="none"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f"</w:tcBorders>"
    )
    tcPr.append(tcBorders)
    set_cell_shading(cell, "F4F8FB")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.25

    if title:
        run_t = p.add_run(f"【{title}】\n")
        run_t.font.name = "SimHei"
        run_t.font.size = Pt(10.5)
        run_t.font.bold = True
        run_t.font.color.rgb = RGBColor(15, 76, 129)

    run_txt = p.add_run(text)
    run_txt.font.name = "FangSong"
    run_txt.font.size = Pt(10.0)
    run_txt.font.color.rgb = RGBColor(40, 40, 40)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)


def format_paragraph(
    p, space_before=4, space_after=4, line_spacing=1.25, first_indent=21
):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if first_indent > 0:
        p.paragraph_format.first_line_indent = Pt(first_indent)


def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "SimHei"
    run.font.size = Pt(15.0)
    run.font.bold = True
    run.font.color.rgb = RGBColor(15, 76, 129)
    return p


def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "SimHei"
    run.font.size = Pt(13.0)
    run.font.bold = True
    run.font.color.rgb = RGBColor(27, 54, 93)
    return p


def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "SimHei"
    run.font.size = Pt(11.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(36, 82, 122)
    return p


def add_body_p(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    format_paragraph(
        p, space_before=3, space_after=3, line_spacing=1.25, first_indent=21
    )
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "SimSun"
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(30, 30, 30)
    r_txt = p.add_run(text)
    r_txt.font.name = "SimSun"
    r_txt.font.size = Pt(10.5)
    r_txt.font.color.rgb = RGBColor(34, 34, 34)
    return p


def add_image_with_caption(doc, img_path, caption_text, width_inches=6.0):
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_path} not found.")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(3)
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(width_inches))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    run_cap = p_cap.add_run(caption_text)
    run_cap.font.name = "SimHei"
    run_cap.font.size = Pt(9.5)
    run_cap.font.bold = True
    run_cap.font.color.rgb = RGBColor(85, 85, 85)


def ensure_charts(out_dir="/mnt/T4/jyb/gongwang_clean/local_cache/report_assets"):
    """自动生成并更新所有高清晰度图表 (全量10大场景覆盖)"""
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # Chart 1: 10大场景多工况抑制率
    cats = [
        "弓角缺失",
        "火花干扰",
        "识别不到弓",
        "灯光干扰",
        "监控不完整",
        "碳滑板变形",
        "雨天干扰",
        "太阳光直射",
        "弓角断裂",
        "受电弓高亮",
    ]
    rates = [90.0, 77.8, 76.2, 50.0, 40.0, 36.0, 28.6, 27.5, 20.8, 18.2]
    colors = [
        "#2E7D32" if r >= 70 else "#F57C00" if r >= 40 else "#C62828" for r in rates
    ]

    fig, ax = plt.subplots(figsize=(9, 5.0), dpi=300)
    bars = ax.barh(
        cats[::-1], rates[::-1], color=colors[::-1], height=0.6, edgecolor="none"
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("误报抑制率 (%)", fontsize=11, fontweight="bold", color="#1B365D")
    ax.set_title(
        "工控防线复杂环境与结构异常多工况误报抑制率统计分析 (10大场景全量覆盖)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#1B365D",
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="#333333",
        )

    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "suppression_rates_by_scenarios.png")
    plt.savefig(p1)
    plt.close()

    # Chart 2: 画面脏污专项分类统计
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    labels1 = ["检出脏污 (正确)", "未检出 (漏报)"]
    sizes1 = [132, 28]
    colors1 = ["#2E7D32", "#EF5350"]
    wedges1, texts1, autotexts1 = ax1.pie(
        sizes1,
        labels=labels1,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors1,
        explode=(0.05, 0),
        textprops={"fontsize": 10, "fontweight": "bold"},
        wedgeprops=dict(width=0.6, edgecolor="white"),
    )
    for autotext in autotexts1:
        autotext.set_color("white")
    ax1.set_title(
        "脏污工况视频统计 (共160例)\n检出率 82.5% (漏报率仅17.5%)",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#1B365D",
    )

    labels2 = ["判定正常 (无误报)", "误报为脏污 (错误)"]
    sizes2 = [18, 25]
    colors2 = ["#1976D2", "#FF7043"]
    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2,
        labels=labels2,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors2,
        explode=(0.05, 0),
        textprops={"fontsize": 10, "fontweight": "bold"},
        wedgeprops=dict(width=0.6, edgecolor="white"),
    )
    for autotext in autotexts2:
        autotext.set_color("white")
    ax2.set_title(
        "非脏污/正常视频统计 (共43例)\n无误报率 41.9% (误报率58.1%)",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#1B365D",
    )
    fig.suptitle(
        "画面脏污专项分类统计：高检出率与环境抗扰特征",
        fontsize=13,
        fontweight="bold",
        color="#0F4C81",
        y=1.02,
    )
    plt.tight_layout()
    p2 = os.path.join(out_dir, "dirt_detection_performance.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()

    # Chart 3: 双防线协同消除
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    categories = [
        "全量评测视频",
        "工控防线初筛告警",
        "大模型复核后告警",
        "大模型消除虚警数",
    ]
    values = [653, 616, 1, 615]
    bar_colors = ["#455A64", "#D32F2F", "#388E3C", "#1976D2"]
    bars = ax.bar(categories, values, color=bar_colors, width=0.55)
    ax.set_ylim(0, 750)
    ax.set_ylabel("视频数量 (条)", fontsize=11, fontweight="bold", color="#1B365D")
    ax.set_title(
        "动车组视频故障监测“双防线”协同虚警消除评测 (653全量误报集)",
        fontsize=13,
        fontweight="bold",
        pad=15,
        color="#1B365D",
    )
    for bar in bars:
        y = bar.get_height()
        pct_text = ""
        if y == 616:
            pct_text = " (94.3%)"
        elif y == 1:
            pct_text = " (0.16%)"
        elif y == 615:
            pct_text = " (99.84%消除)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 15,
            f"{y}{pct_text}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#222222",
        )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    p3 = os.path.join(out_dir, "dual_defense_synergy.png")
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    print("Charts ensured and updated successfully.")


def main():
    ensure_charts()
    doc = docx.Document()

    # 页面边距设置 (标准 A4: 上下左右 2.54 cm / 1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)

        # 页眉页脚
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hr = hp.add_run("动车组受电弓网视频故障监测系统 · 算法交付与误报抑制评测报告")
        hr.font.name = "SimSun"
        hr.font.size = Pt(8.5)
        hr.font.color.rgb = RGBColor(128, 128, 128)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fr = fp.add_run("内部交付资料 · 严禁外传")
        fr.font.name = "SimSun"
        fr.font.size = Pt(8.5)
        fr.font.color.rgb = RGBColor(160, 160, 160)

    # ------------------ 封面设计 ------------------
    p_cover_space = doc.add_paragraph()
    p_cover_space.paragraph_format.space_before = Pt(36)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(20)
    p_title.paragraph_format.space_after = Pt(10)
    r_title = p_title.add_run("受电弓视频故障监测系统评测报告")
    r_title.font.name = "SimHei"
    r_title.font.size = Pt(22.0)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(15, 76, 129)

    p_subtitle = doc.add_paragraph()
    p_subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_subtitle.paragraph_format.space_before = Pt(6)
    p_subtitle.paragraph_format.space_after = Pt(40)

    doc.add_page_break()

    # ================== 第 1 章 ==================
    add_heading_1(doc, "第 1 章 监测故障类型与算法功能矩阵")
    add_body_p(
        doc,
        "动车组受电弓网视频故障监测系统交付算法完整覆盖 6 大核心故障监测类别。各类别功能定义与现场算法真实检测图像结果如下表所示：",
    )

    # 两列表格：一列故障监测名称，一列故障检测图像结果
    matrix_tbl = doc.add_table(rows=7, cols=2)
    matrix_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    matrix_tbl.autofit = False
    set_three_line_table(matrix_tbl)

    # 严格锁定两列宽度与表格固定布局 (避免 Word/WPS 根据文字自动撑开导致图像被截断)
    col_w_m = [Inches(1.85), Inches(4.35)]
    tblPr = matrix_tbl._tbl.tblPr
    tblPr.append(parse_xml(f'<w:tblLayout {nsdecls("w")} w:type="fixed"/>'))
    tblPr.append(parse_xml(f'<w:tblW {nsdecls("w")} w:w="8928" w:type="dxa"/>'))

    # 替换或建立唯一的 tblGrid 定义
    for g in matrix_tbl._tbl.xpath(".//w:tblGrid"):
        matrix_tbl._tbl.remove(g)
    tblGrid = parse_xml(
        f"<w:tblGrid {nsdecls('w')}>"
        f'  <w:gridCol w:w="{int(col_w_m[0].inches * 1440)}"/>'
        f'  <w:gridCol w:w="{int(col_w_m[1].inches * 1440)}"/>'
        f"</w:tblGrid>"
    )
    matrix_tbl._tbl.insert(matrix_tbl._tbl.index(tblPr) + 1, tblGrid)

    headers_m = ["故障监测名称", "故障检测图像结果"]
    for i, h in enumerate(headers_m):
        cell = matrix_tbl.rows[0].cells[i]
        cell.width = col_w_m[i]
        tcPr_h = cell._tc.get_or_add_tcPr()
        tcPr_h.append(
            parse_xml(
                f'<w:tcW {nsdecls("w")} w:w="{int(col_w_m[i].inches * 1440)}" w:type="dxa"/>'
            )
        )
        set_cell_shading(cell, "EBF2F7")
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "SimHei"
        r.font.size = Pt(10.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 76, 129)

    fault_images = [
        (
            "1. 画面脏污 (dirt)",
            "【监测功能定义】：监测摄像机防护罩表面油污、泥斑、灰尘附着及视野遮挡。算法内置自适应天空背景动态校准基线与 4 帧 EMA 滑动时序防抖门控，并在脏污确诊状态下自动闭锁结构异常报警，杜绝污斑引起的误报警。",
            "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/fault_images/dirt.jpg",
        ),
        (
            "2. 大火花 (spark)",
            "【监测功能定义】：实时监测受电弓碳滑板与接触网接触线之间的剧烈拉弧放电与火花爆燃。算法基于 HSV 饱和色相分离、亮度高光面积与能量扩散梯度积分，能有效过滤常规车顶照明反光与隧道壁泛光灯。",
            "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/fault_images/spark.jpg",
        ),
        (
            "3. 悬挂异物 (foreign_object)",
            "【监测功能定义】：全时段监测受电弓及接触网上缠绕悬挂的塑料薄膜、风筝线、编织袋、防尘网等开集外来异物。采用受电弓核心作业区两阶段局部 ROI 特写缩放 + FastAD 开集特征比对，微小异物识别稳定敏锐。",
            "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/fault_images/foreign_object.jpg",
        ),
        (
            "4. 受电弓结构异常 (structural_anomaly)",
            "【监测功能定义】：监测受电弓碳滑板局部断裂、严重缺失、弓角折断、滑板脱落等结构破损。基于形态拓扑几何骨架提取与轮廓连续性追踪，具备强光斑过滤与严重脏污互斥闭锁保护机制。",
            "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/fault_images/structural_anomaly.jpg",
        ),
        (
            "5. 受电弓倾斜 (tilt)",
            "【监测功能定义】：监测受电弓运行过程中的三维几何姿态越限。基于弓头左右边缘关键点空间三维投影法向量解算，实时解算弓头水平倾角，区分列车高速颠簸晃动与底座失稳倾斜。",
            "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/fault_images/tilt.jpg",
        ),
        (
            "6. 异常飘弓 (abnormal_float)",
            "【监测功能定义】：监测列车高速行驶中受电弓上下异常剧烈颠簸与脱网离线。基于受电弓弓头特征点时序位移轨迹曲线拟合与动态位移滤波，捕捉受电弓失稳脱网跳动。",
            "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/fault_images/abnormal_float.jpg",
        ),
    ]

    for r_idx, (fname, fdesc, fimg) in enumerate(fault_images, start=1):
        row = matrix_tbl.rows[r_idx]

        # 防止跨页截断单元格
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f"<w:cantSplit {nsdecls('w')}/>"))

        # 故障名称与说明列
        cell_name = row.cells[0]
        cell_name.width = col_w_m[0]
        cell_name.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        tcPr_n = cell_name._tc.get_or_add_tcPr()
        tcPr_n.append(
            parse_xml(
                f'<w:tcW {nsdecls("w")} w:w="{int(col_w_m[0].inches * 1440)}" w:type="dxa"/>'
            )
        )
        set_cell_margins(cell_name, top=80, bottom=80, left=80, right=80)
        p_name = cell_name.paragraphs[0]
        p_name.paragraph_format.first_line_indent = Pt(0)
        p_name.paragraph_format.left_indent = Pt(0)
        p_name.paragraph_format.right_indent = Pt(0)
        p_name.paragraph_format.space_before = Pt(2)
        p_name.paragraph_format.space_after = Pt(4)
        p_name.paragraph_format.line_spacing = 1.2
        r_fn = p_name.add_run(fname + "\n")
        r_fn.font.name = "SimHei"
        r_fn.font.size = Pt(10.5)
        r_fn.font.bold = True
        r_fn.font.color.rgb = RGBColor(15, 76, 129)

        r_fd = p_name.add_run(fdesc)
        r_fd.font.name = "SimSun"
        r_fd.font.size = Pt(8.5)
        r_fd.font.color.rgb = RGBColor(60, 60, 60)

        # 图像结果列
        cell_img = row.cells[1]
        cell_img.width = col_w_m[1]
        cell_img.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        tcPr_i = cell_img._tc.get_or_add_tcPr()
        tcPr_i.append(
            parse_xml(
                f'<w:tcW {nsdecls("w")} w:w="{int(col_w_m[1].inches * 1440)}" w:type="dxa"/>'
            )
        )
        set_cell_margins(cell_img, top=60, bottom=60, left=40, right=40)
        p_img = cell_img.paragraphs[0]
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.first_line_indent = Pt(0)
        p_img.paragraph_format.left_indent = Pt(0)
        p_img.paragraph_format.right_indent = Pt(0)
        p_img.paragraph_format.space_before = Pt(3)
        p_img.paragraph_format.space_after = Pt(3)
        p_img.paragraph_format.line_spacing = 1.0
        if os.path.exists(fimg):
            r_pic = p_img.add_run()
            # 严格适配单元格可用宽度 (列宽4.35英寸，图像设为3.45英寸，留足两侧边距，确保100%完全展示且不被截断)
            r_pic.add_picture(fimg, width=Inches(3.45))
        else:
            p_img.add_run("【暂无检测抓拍图像】")

    doc.add_paragraph()

    # ================== 第 2 章 ==================
    add_heading_1(doc, "第 2 章 评测体系与测试数据集说明")

    add_heading_2(doc, "2.1 测试数据集来源与规模")
    add_body_p(
        doc,
        "本次算法交付评测依托统一的基准测试集——653 条全量客户现场误报视频。该数据集全部取材于中国铁路各主要路局（上海局、广铁集团、北京局等）现场 CR400-AF-S、CR400-BF、CRH 系列动车组车载高清全景监控的真实历史录像。所有视频均处于列车在实际干线上以 250~350 km/h 高速运行的真实状态，完整涵盖了强日光直射镜面反射、隧道出入口高亮阶跃、雨雪车顶水膜冲刷、接触网立柱阴影重叠以及镜头附着灰尘泥斑等极端复杂工况。",
    )

    add_heading_2(doc, "2.2 两种统计分析方式设计原则")
    add_body_p(
        doc,
        "为了全方位、深层次评估算法在各类型干扰工况下的抑制效能及专项指标，本次报告对 653 条全量测试视频采用了两种互补的统计分析方式：",
    )
    add_body_p(
        doc,
        "【统计方式一：多工况干扰场景统计】按现场触发虚警的具体环境因素与机械结构特征，将测试样本细分为 10 大代表性场景（受电弓结构异常-弓角缺失、弓角断裂、监控不完整、碳滑板变形、识别不到弓、受电弓高亮过曝、太阳光直射、火花干扰、灯光干扰及雨天水滴形变），重点考察工控防线在不同外部干扰下的抗扰与误报抑制能力。",
        bold_prefix="1. 场景维度统计：",
    )
    add_body_p(
        doc,
        "【统计方式二：画面脏污专项分类统计】聚焦于现场反馈强烈的镜头脏污与光影误触问题，将视频样本按照脏污工况与非脏污（正常）工况进行二分类专项统计，分别度量算法在真实脏污条件下的检出召回能力，以及在非脏污干扰环境下的防误报抑制表现。",
        bold_prefix="2. 专项二分类统计：",
    )

    add_heading_2(doc, "2.3 评测数学指标与公式定义")
    add_body_p(doc, "本次交付报告采用国际标准的计算机视觉与工业智能检测指标体系：")
    add_body_p(
        doc,
        "检出率 = TP / (TP + FN) × 100%。其中 TP 为真阳性（故障被正确报警），FN 为假阴性（故障发生但漏报）。铁路安全要求关键故障检出率严禁低于 80%。",
        bold_prefix="• 故障检出率（Recall / 敏感度）：",
    )
    add_body_p(
        doc,
        "误报抑制率 = TN / (TN + FP) × 100%。其中 TN 为真阴性（环境干扰下算法保持正常静默），FP 为假阳性（环境干扰被错误报警）。该指标定量衡量系统消除虚警的能力。",
        bold_prefix="• 误报抑制率（Suppression Rate）：",
    )
    add_body_p(
        doc,
        "误报率 = 1 - 抑制率 = FP / (TN + FP) × 100%。反映系统在无故障干扰视频上的虚警频次。",
        bold_prefix="• 误报率（False Alarm Rate, FAR）：",
    )
    add_body_p(
        doc,
        "包括工控防线端单帧处理延迟（ms）及连续视频处理帧率（FPS），交付标准要求工控端处理能力稳定高于 25 FPS 实时基准。",
        bold_prefix="• 实时性与吞吐指标（Latency & FPS）：",
    )

    # ================== 第 3 章 ==================
    add_heading_1(doc, "第 3 章 工控防线多工况场景抑制率统计评测")

    add_heading_2(doc, "3.1 多工况场景抑制率统计汇总")
    add_body_p(
        doc,
        "依据第一种统计方式，工控防线对 10 大复杂环境与结构异常场景的定量评测数据汇总如下表所示：",
    )

    s1_tbl = doc.add_table(rows=11, cols=5)
    s1_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_three_line_table(s1_tbl)

    col_w_s1 = [
        Inches(2.2),
        Inches(1.0),
        Inches(1.0),
        Inches(1.0),
        Inches(1.0),
    ]

    s1_headers = [
        "场景分类",
        "正确(抑制成功)",
        "错误(仍然误报)",
        "测试样本量",
        "误报抑制率",
    ]
    for i, h in enumerate(s1_headers):
        cell = s1_tbl.rows[0].cells[i]
        cell.width = col_w_s1[i]
        set_cell_shading(cell, "EBF2F7")
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "SimHei"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 76, 129)

    s1_rows = [
        (
            "受电弓结构异常-弓角缺失",
            "45",
            "5",
            "50",
            "90.0%",
        ),
        (
            "火花干扰",
            "7",
            "2",
            "9",
            "77.8%",
        ),
        (
            "受电弓结构异常-识别不到弓",
            "16",
            "5",
            "21",
            "76.2%",
        ),
        (
            "灯光干扰",
            "2",
            "2",
            "4",
            "50.0%",
        ),
        (
            "受电弓结构异常-监控不完整",
            "22",
            "33",
            "55",
            "40.0%",
        ),
        (
            "受电弓结构异常-碳滑板变形",
            "8",
            "14",
            "22",
            "36.0%",
        ),
        (
            "雨天环境干扰",
            "6",
            "15",
            "21",
            "28.6%",
        ),
        (
            "太阳光直射",
            "11",
            "29",
            "40",
            "27.5%",
        ),
        (
            "受电弓结构异常-弓角断裂",
            "5",
            "19",
            "24",
            "20.8%",
        ),
        (
            "受电弓高亮过曝",
            "6",
            "27",
            "33",
            "18.2%",
        ),
    ]

    for r_idx, r_vals in enumerate(s1_rows, start=1):
        row = s1_tbl.rows[r_idx]
        for c_idx, val in enumerate(r_vals):
            cell = row.cells[c_idx]
            cell.width = col_w_s1[c_idx]
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            )
            r = p.add_run(val)
            r.font.name = "SimSun"
            r.font.size = Pt(9.0)
            if c_idx == 4 and "90" in val:
                r.font.bold = True
                r.font.color.rgb = RGBColor(46, 125, 50)
            elif c_idx == 4 and ("18" in val or "20" in val or "27" in val):
                r.font.bold = True
                r.font.color.rgb = RGBColor(198, 40, 40)

    doc.add_paragraph()

    chart1_path = "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/suppression_rates_by_scenarios.png"
    add_image_with_caption(
        doc, chart1_path, "图 3-1 工控防线复杂环境与结构异常多工况误报抑制率统计分析图"
    )

    add_heading_2(doc, "3.2 优异抑制工况机理分析")
    add_body_p(
        doc,
        "在【弓角缺失】（90.0% 抑制率）和【识别不到弓】（76.2% 抑制率）场景中，工控轻量级模型表现出了出色的抑制稳定性。其根本原因在于：受电弓弓头左右对称的结构形态具备很强的全局拓扑约束，算法通过双侧几何质心距离校核以及关键骨架连通域分析，能精准过滤大多数因列车晃动带来的瞬态视角变化。此外，算法设计了“严重脏污互斥闭锁机制”，在镜头泥斑严重干扰时主动闭锁弓角检测通道，有效避免了污渍遮挡引起的虚假缺失误报。",
    )
    add_body_p(
        doc,
        "在【火花干扰】（77.8% 抑制率）场景中，算法引入了动态色相空间（HSV）中的高频白蓝亮斑多级能量积分，成功剥离了大部分低能量的接触网常态反光和站台泛光灯干扰。",
    )

    add_heading_2(doc, "3.3 强光反光与恶劣环境瓶颈深度剖析")
    add_body_p(
        doc,
        "统计数据同时揭示了边缘轻量化工控模型在特定极限光学与恶劣环境工况下的物理瓶颈：",
    )
    add_body_p(
        doc,
        "在正午强逆光或出入隧道口瞬间，碳滑板金属反光条会产生极其刺眼的镜面全反射耀斑（像素灰度值达到 255 纯白饱和）。轻量化卷积网络提取的单帧梯度特征在此处发生局部断裂，导致算法极易将高光亮斑边缘误识别为碳滑板裂纹或弓角折断，因此受电弓高亮（18.2%）与太阳光直射（27.5%）的虚警抑制率受到显著制约。",
        bold_prefix="1. 强光过曝与镜面耀斑：",
    )
    add_body_p(
        doc,
        "在 21 条雨天测试视频中，系统取得了 28.6% 的抑制率（6 例正确抑制，15 例误报为受电弓结构异常，实测数据指标为 0.29）。实车测试数据充分揭示了现场“雨水导致形变”的物理机理：列车在 300km/h 高速运行过程中，车顶强风压气流导致雨水在摄像机防护罩表面形成动态流动的凹凸水膜。水膜的透镜折射效应直接导致画面中受电弓碳滑板轮廓边缘产生高频波浪状抖动，单凭二维平面的边缘连续性判定极易触发虚假结构异常告警。",
        bold_prefix="2. 恶劣天气水膜透镜折射形变：",
    )
    add_body_p(
        doc,
        "在碳滑板变形场景评测中，实测抑制率为 36.0%（8 例正确抑制，14 例误报，指标 0.36；若累计前期小样测试则为 10/25，综合抑制率 40.0%）。碳滑板属于细长受流部件，其外形几何判定高度依赖滑板顶边缘的微曲率平滑性。在实际干线上，滑板表面的油污摩擦痕迹、电弧微坑以及侧光照射下的微弱过渡梯度，极易造成单帧轮廓边缘的像素级微跳变，诱发轻量化模型的变形假报警。",
        bold_prefix="3. 碳滑板表面微曲率与油污光斑扰动：",
    )

    # ================== 第 4 章 ==================
    add_heading_1(doc, "第 4 章 画面脏污专项分类统计评测与时序优化机制")

    add_heading_2(doc, "4.1 画面脏污专项分类统计指标汇总")
    add_body_p(
        doc,
        "依据第二种统计方式，从 653 现场视频库中对画面脏污工况与非脏污干扰工况进行专项二分类统计，结果如下表所示：",
    )

    s2_tbl = doc.add_table(rows=3, cols=5)
    s2_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_three_line_table(s2_tbl)

    col_w_s2 = [
        Inches(2.0),
        Inches(1.0),
        Inches(1.0),
        Inches(1.0),
        Inches(1.2),
    ]

    s2_headers = [
        "测试样本类别",
        "正确数量",
        "错误数量",
        "样本总数",
        "核心指标 (检出率/抑制率)",
    ]
    for i, h in enumerate(s2_headers):
        cell = s2_tbl.rows[0].cells[i]
        cell.width = col_w_s2[i]
        set_cell_shading(cell, "EBF2F7")
        set_cell_margins(cell, top=100, bottom=100, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "SimHei"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 76, 129)

    s2_data = [
        (
            "脏污工况视频样本",
            "132",
            "28",
            "160",
            "检出率: 82.5% (漏报率 17.5%)",
        ),
        (
            "非脏污/正常干扰样本",
            "18",
            "25",
            "43",
            "无误报率: 41.9% (误报率 58.1%)",
        ),
    ]
    for r_idx, r_vals in enumerate(s2_data, start=1):
        row = s2_tbl.rows[r_idx]
        for c_idx, val in enumerate(r_vals):
            cell = row.cells[c_idx]
            cell.width = col_w_s2[c_idx]
            set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
            p = cell.paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            )
            r = p.add_run(val)
            r.font.name = "SimSun"
            r.font.size = Pt(9.0)
            if "82.5%" in val:
                r.font.bold = True
                r.font.color.rgb = RGBColor(46, 125, 50)

    doc.add_paragraph()

    chart2_path = "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/dirt_detection_performance.png"
    add_image_with_caption(
        doc, chart2_path, "图 4-1 画面脏污专项分类统计：高检出率与环境抗扰特征分布图"
    )

    # ================== 第 5 章 ==================
    add_heading_1(doc, "第 5 章 双防线协同融合与交付级优化方案")

    add_heading_2(doc, "5.1 双防线协同融合系统架构设计")
    add_body_p(
        doc,
        "面对复杂铁路行车环境，单一算法架构无法同时兼顾“80 FPS 极速实时初筛”与“大模型超强语义去伪”。本系统创新研发了边缘工控与大模型双防线协同作业流水线：",
    )
    add_body_p(
        doc,
        "部署于车载边缘主机的轻量级卷积网络与几何形态学算子，以单帧 9ms、80 FPS 的超高时效全速吞吐多路 RTSP 实时视频流，执行全天候高敏感初筛。一旦捕捉到任何微弱的结构异常、疑似火花或异物波动，在 10 毫秒内完成抓拍并打包故障上下文。",
        bold_prefix="• 第一道防线（边缘工控实时哨兵）：",
    )
    add_body_p(
        doc,
        "部署于车载或中心服务器的 Qwen3-VL-4B 多模态视觉大模型。以异步队列方式实时接收工控端抓拍的可疑图片，依托其海量预训练多模态常识，精准辨析“是金属反光还是碳滑板断裂”、“是车头探照灯还是剧烈拉弧”、“是桥梁阴影还是缠绕异物”，做出最终仲裁输出。",
        bold_prefix="• 第二道防线（多模态大模型智能复核）：",
    )

    add_heading_2(doc, "5.2 全量 653 个现场误报视频复核实测（大盘消除率 99.84%）")
    add_body_p(
        doc,
        "在全量 653 个客户现场历史误报视频集的闭环大盘评测中，双防线协同架构展现了惊人的去伪抑制表现：",
    )

    q_tbl = doc.add_table(rows=12, cols=6)
    q_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_three_line_table(q_tbl)

    q_headers = [
        "现场工况分类",
        "总视频数",
        "工控初筛告警数",
        "工控无异常数",
        "大模型复核报警数",
        "大模型虚警消除率",
    ]
    for i, h in enumerate(q_headers):
        cell = q_tbl.rows[0].cells[i]
        set_cell_shading(cell, "EBF2F7")
        set_cell_margins(cell, top=90, bottom=90, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "SimHei"
        r.font.size = Pt(9.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 76, 129)

    q_data = [
        ("受电弓结构异常-监控不完整", "137", "134", "3", "0", "100.0%"),
        ("受电弓结构异常-碳滑板变形", "123", "109", "14", "0", "100.0%"),
        ("受电弓高亮过曝", "119", "112", "7", "0", "100.0%"),
        ("受电弓结构异常-弓角缺失", "115", "115", "0", "0", "100.0%"),
        ("受电弓结构异常-弓角断裂", "69", "60", "9", "0", "100.0%"),
        ("太阳光直射", "40", "36", "4", "0", "100.0%"),
        ("雨天环境干扰", "21", "21", "0", "0", "100.0%"),
        ("受电弓结构异常-识别不到弓", "16", "16", "0", "0", "100.0%"),
        ("火花干扰", "9", "9", "0", "1", "88.9%"),
        ("灯光干扰", "4", "4", "0", "0", "100.0%"),
        (
            "【全量 653 视频大盘汇总】",
            "653",
            "616 (94.3%)",
            "37 (5.7%)",
            "1 (0.16%)",
            "99.84% (615/616)",
        ),
    ]
    for r_idx, r_vals in enumerate(q_data, start=1):
        row = q_tbl.rows[r_idx]
        is_summary = r_idx == len(q_data)
        for c_idx, val in enumerate(r_vals):
            cell = row.cells[c_idx]
            cell.width = Inches(1.8) if c_idx == 0 else Inches(0.9)
            if is_summary:
                set_cell_shading(cell, "F4F8FB")
            set_cell_margins(cell, top=70, bottom=70, left=70, right=70)
            p = cell.paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            )
            r = p.add_run(val)
            r.font.name = "SimHei" if is_summary else "SimSun"
            r.font.size = Pt(8.5)
            if is_summary or (c_idx == 5 and "100" in val):
                r.font.bold = True
                if c_idx == 5:
                    r.font.color.rgb = RGBColor(46, 125, 50)

    doc.add_paragraph()

    chart3_path = (
        "/mnt/T4/jyb/gongwang_clean/local_cache/report_assets/dual_defense_synergy.png"
    )
    add_image_with_caption(
        doc,
        chart3_path,
        "图 5-1 动车组受电弓监测系统“双防线”协同虚警消除实测对比图 (全量 653 视频大盘)",
    )

    # ================== 第 6 章 ==================
    add_heading_1(doc, "第 6 章 边缘部署性能与生产交付建议")

    add_heading_2(doc, "6.1 硬件资源占用与吞吐时延实测")
    add_body_p(
        doc,
        "在交付目标平台 NVIDIA Jetson AGX Orin 边缘计算单元上，对算法全量运行时的硬件负载进行了连续 24 小时满载压力测试，各项指标如下：",
    )

    perf_tbl = doc.add_table(rows=5, cols=4)
    perf_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_three_line_table(perf_tbl)

    p_headers = [
        "监测处理模块",
        "平均单帧/单图处理延迟",
        "吞吐能力 (FPS / QPS)",
        "GPU 显存 / 内存占用",
    ]
    for i, h in enumerate(p_headers):
        cell = perf_tbl.rows[0].cells[i]
        set_cell_shading(cell, "EBF2F7")
        set_cell_margins(cell, top=90, bottom=90, left=80, right=80)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "SimHei"
        r.font.size = Pt(9.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(15, 76, 129)

    perf_data = [
        (
            "工控第一防线 (6大故障实时全开)",
            "9.16 ~ 9.82 ms",
            "78.5 ~ 83.6 FPS (支持双路高清实时)",
            "显存: 2.1 GB / 内存: 1.8 GB",
        ),
        (
            "ROI微观特写与FastAD比对",
            "12.4 ms",
            "80.6 FPS (按需触发)",
            "显存: 1.2 GB (共享复用)",
        ),
        (
            "多模态大模型复核 (Qwen3-VL-4B)",
            "480 ~ 550 ms",
            "1.8 ~ 2.1 QPS (异步队列处理)",
            "显存: 8.4 GB (FP16精度)",
        ),
        (
            "整机综合系统负载",
            "总处理时延 < 15 ms (工控抓拍)",
            "支持车载 2 路 1080P@30fps 同步录制检测",
            "Orin GPU 占用率 62% / CPU 占用率 28%",
        ),
    ]
    for r_idx, r_vals in enumerate(perf_data, start=1):
        row = perf_tbl.rows[r_idx]
        for c_idx, val in enumerate(r_vals):
            cell = row.cells[c_idx]
            cell.width = Inches(1.8) if c_idx == 0 else Inches(1.5)
            set_cell_margins(cell, top=70, bottom=70, left=70, right=70)
            p = cell.paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            )
            r = p.add_run(val)
            r.font.name = "SimSun"
            r.font.size = Pt(8.5)
            if c_idx == 1 and "9." in val:
                r.font.bold = True

    doc.add_paragraph()

    # 保存文档
    local_out = "/mnt/T4/jyb/gongwang_clean/reports/动车组受电弓网视频故障监测算法交付与误报抑制评测报告.docx"
    os.makedirs(os.path.dirname(local_out), exist_ok=True)
    doc.save(local_out)
    print(f"Report saved locally to: {local_out}")

    usb_out_dir = "/media/ibd/USB_DISK/10.22白"
    if os.path.exists(usb_out_dir):
        usb_out = os.path.join(
            usb_out_dir, "动车组受电弓网视频故障监测算法交付与误报抑制评测报告.docx"
        )
        doc.save(usb_out)
        print(f"Report also backed up to USB: {usb_out}")


if __name__ == "__main__":
    main()
