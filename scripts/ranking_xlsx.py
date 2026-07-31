"""Small dependency-free XLSX writer for benchmark ranking workbooks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


HEADERS = (
    "Rank",
    "Model",
    "Mean AUC-PR",
    "Std AUC-PR",
    "Mean AUC-ROC",
    "Std AUC-ROC",
    "Mean Fit (s)",
    "Mean Inference (s)",
    "Successful Runs",
    "Status",
)


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _string_cell(reference: str, value: Any, style: int) -> str:
    text = escape(str(value))
    return (
        f'<c r="{reference}" s="{style}" t="inlineStr">'
        f"<is><t>{text}</t></is></c>"
    )


def _number_cell(reference: str, value: Any, style: int) -> str:
    if value is None:
        return f'<c r="{reference}" s="{style}"/>'
    return f'<c r="{reference}" s="{style}"><v>{float(value):.12g}</v></c>'


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="2">
    <numFmt numFmtId="164" formatCode="0.000000"/>
    <numFmt numFmtId="165" formatCode="0.00%"/>
  </numFmts>
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="14"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="7">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFD966"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9E1F2"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF4B183"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E2F3"/></left><right style="thin"><color rgb="FFD9E2F3"/></right><top style="thin"><color rgb="FFD9E2F3"/></top><bottom style="thin"><color rgb="FFD9E2F3"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="13">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="left"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="left"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"><alignment horizontal="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"><alignment horizontal="left"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0"><alignment horizontal="right"/></xf>
    <xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="left"/></xf>
    <xf numFmtId="0" fontId="0" fillId="4" borderId="1" xfId="0"><alignment horizontal="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0"><alignment horizontal="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0"><alignment horizontal="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"><alignment horizontal="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def _worksheet_xml(
    dataset_name: str,
    metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> str:
    last_row = 4 + len(rows)
    sheet_rows = [
        '<row r="1" ht="24" customHeight="1">'
        + _string_cell("A1", f"MODEL RANKING - {dataset_name}", 1)
        + "</row>"
    ]
    metadata_cells = [
        _string_cell("A2", "Samples", 2),
        _number_cell("B2", metadata["samples"], 3),
        _string_cell("C2", "Features", 2),
        _number_cell("D2", metadata["features"], 3),
        _string_cell("E2", "Anomalies", 2),
        _number_cell("F2", metadata["anomalies"], 3),
        _string_cell("G2", "Anomaly ratio", 2),
        _number_cell("H2", metadata["anomaly_ratio"], 8),
        _string_cell("I2", "Ranking metric", 2),
        _string_cell("J2", "Mean AUC-PR", 3),
    ]
    sheet_rows.append('<row r="2">' + "".join(metadata_cells) + "</row>")
    header_cells = [
        _string_cell(f"{_column_name(index)}4", header, 4)
        for index, header in enumerate(HEADERS, start=1)
    ]
    sheet_rows.append('<row r="4" ht="30" customHeight="1">' + "".join(header_cells) + "</row>")

    value_keys = (
        "Rank",
        "Model",
        "Mean AUC-PR",
        "Std AUC-PR",
        "Mean AUC-ROC",
        "Std AUC-ROC",
        "Mean Fit (s)",
        "Mean Inference (s)",
        "Successful Runs",
        "Status",
    )
    numeric_keys = {
        "Rank",
        "Mean AUC-PR",
        "Std AUC-PR",
        "Mean AUC-ROC",
        "Std AUC-ROC",
        "Mean Fit (s)",
        "Mean Inference (s)",
        "Successful Runs",
    }
    for row_number, row in enumerate(rows, start=5):
        cells = []
        for column_number, key in enumerate(value_keys, start=1):
            reference = f"{_column_name(column_number)}{row_number}"
            value = row.get(key)
            if key == "Rank":
                rank = int(value)
                style = {1: 9, 2: 10, 3: 11}.get(rank, 5)
                cells.append(_number_cell(reference, rank, style))
            elif key in numeric_keys:
                style = 5 if key == "Successful Runs" else 7
                cells.append(_number_cell(reference, value, style))
            else:
                cells.append(_string_cell(reference, value, 6 if key == "Model" else 12))
        sheet_rows.append(f'<row r="{row_number}">' + "".join(cells) + "</row>")

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:J{last_row}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="4" topLeftCell="A5" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="1" width="8" customWidth="1"/>
    <col min="2" max="2" width="18" customWidth="1"/>
    <col min="3" max="6" width="16" customWidth="1"/>
    <col min="7" max="8" width="20" customWidth="1"/>
    <col min="9" max="9" width="17" customWidth="1"/>
    <col min="10" max="10" width="12" customWidth="1"/>
  </cols>
  <sheetData>{''.join(sheet_rows)}</sheetData>
  <mergeCells count="1"><mergeCell ref="A1:J1"/></mergeCells>
  <autoFilter ref="A4:J{last_row}"/>
  <pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>
</worksheet>"""


def write_ranking_workbook(
    path: Path,
    dataset_name: str,
    metadata: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> Path:
    """Write one formatted ranking workbook without optional Excel packages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    root_relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    workbook = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Ranking" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    workbook_relationships = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>ADBench</dc:creator><cp:lastModifiedBy>ADBench</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>ADBench</Application></Properties>"""

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_relationships)
        archive.writestr("xl/styles.xml", _styles_xml())
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            _worksheet_xml(dataset_name, metadata, rows),
        )
    return path
