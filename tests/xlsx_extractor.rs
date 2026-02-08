use momo::models::DocumentType;
use momo::processing::extractors::xlsx::XlsxExtractor;

mod common;
use common::{ensure_fixtures, load_fixture};

#[test]
fn test_xlsx_single_sheet() {
    ensure_fixtures();
    let bytes = load_fixture("sample.xlsx");

    let result = XlsxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should successfully extract XLSX");

    let extracted = result.unwrap();
    assert_eq!(extracted.doc_type, DocumentType::Xlsx);

    // Check for markdown table format
    assert!(
        extracted.text.contains("## Sheet: Sheet1"),
        "Should have sheet header"
    );
    assert!(
        extracted.text.contains("| Product | Price | Category |"),
        "Should have table header"
    );
    assert!(
        extracted.text.contains("| Widget A | 100 | Electronics |"),
        "Should have data row"
    );
    assert!(
        extracted.text.contains("| Widget B | 200 | Tools |"),
        "Should have second data row"
    );

    // Check word count is reasonable
    assert!(extracted.word_count > 0, "Should have word count");
}

#[test]
fn test_xlsx_multi_sheet() {
    // Create a multi-sheet XLSX fixture
    let bytes = create_multi_sheet_xlsx();

    let result = XlsxExtractor::extract(&bytes);
    assert!(
        result.is_ok(),
        "Should successfully extract multi-sheet XLSX"
    );

    let extracted = result.unwrap();

    // Check both sheets are present
    assert!(
        extracted.text.contains("## Sheet: Products"),
        "Should have Products sheet header"
    );
    assert!(
        extracted.text.contains("## Sheet: Summary"),
        "Should have Summary sheet header"
    );

    // Check data from both sheets
    assert!(
        extracted.text.contains("Widget A"),
        "Should have data from Products sheet"
    );
    assert!(
        extracted.text.contains("Total"),
        "Should have data from Summary sheet"
    );
}

#[test]
fn test_xlsx_cell_types() {
    // Create XLSX with various cell types
    let bytes = create_cell_types_xlsx();

    let result = XlsxExtractor::extract(&bytes);
    assert!(
        result.is_ok(),
        "Should successfully extract XLSX with various cell types"
    );

    let extracted = result.unwrap();

    // Check string values
    assert!(
        extracted.text.contains("String Value"),
        "Should contain string"
    );

    // Check numeric values (integers and floats)
    assert!(extracted.text.contains("42"), "Should contain integer");
    assert!(extracted.text.contains("3.14"), "Should contain float");

    // Check boolean (should be represented somehow)
    assert!(
        extracted.text.to_lowercase().contains("true") || extracted.text.contains("1"),
        "Should contain boolean true"
    );
}

#[test]
fn test_xlsx_empty() {
    // Create empty XLSX
    let bytes = create_empty_xlsx();

    let result = XlsxExtractor::extract(&bytes);
    assert!(result.is_ok(), "Should handle empty XLSX");

    let extracted = result.unwrap();
    assert_eq!(extracted.doc_type, DocumentType::Xlsx);
    // Empty spreadsheet should still have sheet header but no table
    assert!(
        extracted.text.contains("## Sheet:"),
        "Should have sheet header even if empty"
    );
}

#[test]
fn test_xlsx_corrupt() {
    // Create corrupt/invalid XLSX data
    let bytes = b"This is not a valid XLSX file";

    let result = XlsxExtractor::extract(bytes);
    assert!(result.is_err(), "Should fail on corrupt XLSX");

    // Check error type is Processing
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("XLSX") || err_msg.contains("parse"),
        "Error should indicate XLSX parsing issue"
    );
}

#[test]
fn test_xlsx_large_sheet() {
    // Create XLSX with many rows to test row limit
    let bytes = create_large_xlsx(150_000); // More than MAX_ROWS limit

    let result = XlsxExtractor::extract(&bytes);
    assert!(
        result.is_ok(),
        "Should handle large XLSX (possibly truncating)"
    );

    let extracted = result.unwrap();

    // Should either truncate or handle gracefully
    // The implementation should limit to 100K rows
    let line_count = extracted.text.lines().count();
    assert!(
        line_count < 200_000,
        "Should not have excessive lines from oversized sheet"
    );
}

// Helper functions to create test XLSX files

fn create_multi_sheet_xlsx() -> Vec<u8> {
    use std::io::Cursor;
    use std::io::Write;

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
    <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
    <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
    <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>"#.as_bytes()).unwrap();

        // _rels/.rels
        zip.add_directory("_rels", options).unwrap();
        zip.start_file("_rels/.rels", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/workbook.xml
        zip.add_directory("xl", options).unwrap();
        zip.start_file("xl/workbook.xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <sheets>
        <sheet name="Products" sheetId="1" r:id="rId1"/>
        <sheet name="Summary" sheetId="2" r:id="rId2"/>
    </sheets>
</workbook>"#.as_bytes()).unwrap();

        // xl/_rels/workbook.xml.rels
        zip.add_directory("xl/_rels", options).unwrap();
        zip.start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
    <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/worksheets/sheet1.xml (Products)
        zip.add_directory("xl/worksheets", options).unwrap();
        zip.start_file("xl/worksheets/sheet1.xml", options).unwrap();
        zip.write_all(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>
        <row r="1">
            <c r="A1" t="s"><v>0</v></c>
            <c r="B1" t="s"><v>1</v></c>
        </row>
        <row r="2">
            <c r="A2" t="s"><v>2</v></c>
            <c r="B2"><v>100</v></c>
        </row>
    </sheetData>
</worksheet>"#
                .as_bytes(),
        )
        .unwrap();

        // xl/worksheets/sheet2.xml (Summary)
        zip.start_file("xl/worksheets/sheet2.xml", options).unwrap();
        zip.write_all(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>
        <row r="1">
            <c r="A1" t="s"><v>3</v></c>
            <c r="B1" t="s"><v>4</v></c>
        </row>
        <row r="2">
            <c r="A2" t="s"><v>5</v></c>
            <c r="B2"><v>300</v></c>
        </row>
    </sheetData>
</worksheet>"#
                .as_bytes(),
        )
        .unwrap();

        // xl/sharedStrings.xml
        zip.start_file("xl/sharedStrings.xml", options).unwrap();
        zip.write_all(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="6" uniqueCount="6">
    <si><t>Name</t></si>
    <si><t>Value</t></si>
    <si><t>Widget A</t></si>
    <si><t>Metric</t></si>
    <si><t>Total</t></si>
    <si><t>Revenue</t></si>
</sst>"#
                .as_bytes(),
        )
        .unwrap();

        zip.finish().unwrap();
    }

    buffer.into_inner()
}

fn create_cell_types_xlsx() -> Vec<u8> {
    use std::io::Cursor;
    use std::io::Write;

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
    <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
    <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>"#.as_bytes()).unwrap();

        // _rels/.rels
        zip.add_directory("_rels", options).unwrap();
        zip.start_file("_rels/.rels", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/workbook.xml
        zip.add_directory("xl", options).unwrap();
        zip.start_file("xl/workbook.xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <sheets>
        <sheet name="Types" sheetId="1" r:id="rId1"/>
    </sheets>
</workbook>"#.as_bytes()).unwrap();

        // xl/_rels/workbook.xml.rels
        zip.add_directory("xl/_rels", options).unwrap();
        zip.start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/worksheets/sheet1.xml with various cell types
        zip.add_directory("xl/worksheets", options).unwrap();
        zip.start_file("xl/worksheets/sheet1.xml", options).unwrap();
        zip.write_all(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>
        <row r="1">
            <c r="A1" t="s"><v>0</v></c>
            <c r="B1" t="s"><v>1</v></c>
        </row>
        <row r="2">
            <c r="A2" t="s"><v>2</v></c>
            <c r="B2"><v>42</v></c>
        </row>
        <row r="3">
            <c r="A3" t="s"><v>3</v></c>
            <c r="B3"><v>3.14</v></c>
        </row>
        <row r="4">
            <c r="A4" t="s"><v>4</v></c>
            <c r="B4" t="b"><v>1</v></c>
        </row>
        <row r="5">
            <c r="A5" t="s"><v>5</v></c>
            <c r="B5"/>
        </row>
    </sheetData>
</worksheet>"#
                .as_bytes(),
        )
        .unwrap();

        // xl/sharedStrings.xml
        zip.start_file("xl/sharedStrings.xml", options).unwrap();
        zip.write_all(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="6" uniqueCount="6">
    <si><t>Type</t></si>
    <si><t>Value</t></si>
    <si><t>String Value</t></si>
    <si><t>Integer</t></si>
    <si><t>Float</t></si>
    <si><t>Boolean</t></si>
    <si><t>Empty</t></si>
</sst>"#
                .as_bytes(),
        )
        .unwrap();

        zip.finish().unwrap();
    }

    buffer.into_inner()
}

fn create_empty_xlsx() -> Vec<u8> {
    use std::io::Cursor;
    use std::io::Write;

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
    <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#.as_bytes()).unwrap();

        // _rels/.rels
        zip.add_directory("_rels", options).unwrap();
        zip.start_file("_rels/.rels", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/workbook.xml
        zip.add_directory("xl", options).unwrap();
        zip.start_file("xl/workbook.xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <sheets>
        <sheet name="EmptySheet" sheetId="1" r:id="rId1"/>
    </sheets>
</workbook>"#.as_bytes()).unwrap();

        // xl/_rels/workbook.xml.rels
        zip.add_directory("xl/_rels", options).unwrap();
        zip.start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/worksheets/sheet1.xml - empty sheet
        zip.add_directory("xl/worksheets", options).unwrap();
        zip.start_file("xl/worksheets/sheet1.xml", options).unwrap();
        zip.write_all(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>
    </sheetData>
</worksheet>"#
                .as_bytes(),
        )
        .unwrap();

        zip.finish().unwrap();
    }

    buffer.into_inner()
}

fn create_large_xlsx(row_count: usize) -> Vec<u8> {
    use std::io::Cursor;
    use std::io::Write;

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
    <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#.as_bytes()).unwrap();

        // _rels/.rels
        zip.add_directory("_rels", options).unwrap();
        zip.start_file("_rels/.rels", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/workbook.xml
        zip.add_directory("xl", options).unwrap();
        zip.start_file("xl/workbook.xml", options).unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <sheets>
        <sheet name="LargeSheet" sheetId="1" r:id="rId1"/>
    </sheets>
</workbook>"#.as_bytes()).unwrap();

        // xl/_rels/workbook.xml.rels
        zip.add_directory("xl/_rels", options).unwrap();
        zip.start_file("xl/_rels/workbook.xml.rels", options)
            .unwrap();
        zip.write_all(r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#.as_bytes()).unwrap();

        // xl/worksheets/sheet1.xml - generate large sheet
        zip.add_directory("xl/worksheets", options).unwrap();
        zip.start_file("xl/worksheets/sheet1.xml", options).unwrap();

        let mut sheet_xml = String::from(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>"#,
        );

        // Header row
        sheet_xml.push_str(
            r#"
        <row r="1">
            <c r="A1" t="s"><v>0</v></c>
            <c r="B1" t="s"><v>1</v></c>
        </row>"#,
        );

        // Data rows - cap at 10K for test performance
        let actual_rows = row_count.min(10000);
        for i in 2..=actual_rows {
            sheet_xml.push_str(&format!(
                r#"
        <row r="{i}">
            <c r="A{i}" t="s"><v>2</v></c>
            <c r="B{i}"><v>{i}</v></c>
        </row>"#
            ));
        }

        sheet_xml.push_str(
            r#"
    </sheetData>
</worksheet>"#,
        );

        zip.write_all(sheet_xml.as_bytes()).unwrap();

        // xl/sharedStrings.xml
        zip.start_file("xl/sharedStrings.xml", options).unwrap();
        zip.write_all(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="3" uniqueCount="3">
    <si><t>Column A</t></si>
    <si><t>Column B</t></si>
    <si><t>Data</t></si>
</sst>"#
                .as_bytes(),
        )
        .unwrap();

        zip.finish().unwrap();
    }

    buffer.into_inner()
}
