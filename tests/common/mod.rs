use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Get the path to a fixture file
pub fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Load a fixture file as bytes
pub fn load_fixture(name: &str) -> Vec<u8> {
    let path = fixture_path(name);
    fs::read(&path).unwrap_or_else(|e| panic!("Failed to load fixture '{}': {}", name, e))
}

/// Ensure all fixture files exist, generating them if necessary
pub fn ensure_fixtures() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures");

    fs::create_dir_all(&fixtures_dir).expect("Failed to create fixtures directory");

    // Generate CSV fixture
    let csv_path = fixtures_dir.join("sample.csv");
    if !csv_path.exists() {
        generate_csv_fixture(&csv_path);
    }

    // Generate DOCX fixture
    let docx_path = fixtures_dir.join("sample.docx");
    if !docx_path.exists() {
        generate_docx_fixture(&docx_path);
    }

    // Generate XLSX fixture
    let xlsx_path = fixtures_dir.join("sample.xlsx");
    if !xlsx_path.exists() {
        generate_xlsx_fixture(&xlsx_path);
    }

    // Generate PPTX fixture
    let pptx_path = fixtures_dir.join("sample.pptx");
    if !pptx_path.exists() {
        generate_pptx_fixture(&pptx_path);
    }
}

fn generate_csv_fixture(path: &Path) {
    let csv_content = r#"Name,Age,City,Occupation
Alice,30,New York,Engineer
Bob,25,Los Angeles,Designer
Charlie,35,Chicago,Manager
Diana,28,Seattle,Developer
"#;
    fs::write(path, csv_content).expect("Failed to write CSV fixture");
}

fn generate_docx_fixture(path: &Path) {
    use docx_rs::*;

    let docx = Docx::new()
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text("Hello World").bold()))
        .add_paragraph(
            Paragraph::new().add_run(Run::new().add_text("This is a test document for Momo.")),
        )
        .add_paragraph(
            Paragraph::new()
                .add_run(Run::new().add_text("It contains multiple paragraphs of text.")),
        );

    let mut buffer = Cursor::new(Vec::new());
    docx.build().pack(&mut buffer).expect("Failed to pack DOCX");
    fs::write(path, buffer.into_inner()).expect("Failed to write DOCX fixture");
}

fn generate_xlsx_fixture(path: &Path) {
    use std::io::Write;
    use zip::write::FileOptions;
    use zip::CompressionMethod;

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options: FileOptions<zip::write::ExtendedFileOptions> = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        // [Content_Types].xml
        zip.start_file("[Content_Types].xml", options.clone())
            .unwrap();
        zip.write_all(CONTENT_TYPES_XLSX.as_bytes()).unwrap();

        // _rels/.rels
        zip.add_directory("_rels", options.clone()).unwrap();
        zip.start_file("_rels/.rels", options.clone()).unwrap();
        zip.write_all(RELS_XLSX.as_bytes()).unwrap();

        // xl/workbook.xml
        zip.add_directory("xl", options.clone()).unwrap();
        zip.start_file("xl/workbook.xml", options.clone()).unwrap();
        zip.write_all(WORKBOOK_XML.as_bytes()).unwrap();

        // xl/_rels/workbook.xml.rels
        zip.add_directory("xl/_rels", options.clone()).unwrap();
        zip.start_file("xl/_rels/workbook.xml.rels", options.clone())
            .unwrap();
        zip.write_all(WORKBOOK_RELS.as_bytes()).unwrap();

        // xl/worksheets/sheet1.xml
        zip.add_directory("xl/worksheets", options.clone()).unwrap();
        zip.start_file("xl/worksheets/sheet1.xml", options.clone())
            .unwrap();
        zip.write_all(SHEET1_XML.as_bytes()).unwrap();

        // xl/sharedStrings.xml
        zip.start_file("xl/sharedStrings.xml", options.clone())
            .unwrap();
        zip.write_all(SHARED_STRINGS_XML.as_bytes()).unwrap();

        zip.finish().unwrap();
    }

    fs::write(path, buffer.into_inner()).expect("Failed to write XLSX fixture");
}

// XLSX XML content
const CONTENT_TYPES_XLSX: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
    <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
    <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>"#;

const RELS_XLSX: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#;

const WORKBOOK_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <sheets>
        <sheet name="Sheet1" sheetId="1" r:id="rId1"/>
    </sheets>
</workbook>"#;

const WORKBOOK_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>"#;

const SHEET1_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>
        <row r="1">
            <c r="A1" t="s"><v>0</v></c>
            <c r="B1" t="s"><v>1</v></c>
            <c r="C1" t="s"><v>2</v></c>
        </row>
        <row r="2">
            <c r="A2" t="s"><v>3</v></c>
            <c r="B2"><v>100</v></c>
            <c r="C2" t="s"><v>4</v></c>
        </row>
        <row r="3">
            <c r="A3" t="s"><v>5</v></c>
            <c r="B3"><v>200</v></c>
            <c r="C3" t="s"><v>6</v></c>
        </row>
    </sheetData>
</worksheet>"#;

const SHARED_STRINGS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="7" uniqueCount="7">
    <si><t>Product</t></si>
    <si><t>Price</t></si>
    <si><t>Category</t></si>
    <si><t>Widget A</t></si>
    <si><t>Electronics</t></si>
    <si><t>Widget B</t></si>
    <si><t>Tools</t></si>
</sst>"#;

fn generate_pptx_fixture(path: &Path) {
    use std::io::Write;
    use zip::write::FileOptions;
    use zip::CompressionMethod;

    let mut buffer = Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buffer);
        let options: FileOptions<zip::write::ExtendedFileOptions> = FileOptions::default()
            .compression_method(CompressionMethod::Deflated)
            .unix_permissions(0o644);

        zip.start_file("[Content_Types].xml", options.clone())
            .unwrap();
        zip.write_all(CONTENT_TYPES_PPTX.as_bytes()).unwrap();

        zip.add_directory("_rels", options.clone()).unwrap();
        zip.start_file("_rels/.rels", options.clone()).unwrap();
        zip.write_all(RELS_PPTX.as_bytes()).unwrap();

        zip.add_directory("ppt", options.clone()).unwrap();
        zip.start_file("ppt/presentation.xml", options.clone())
            .unwrap();
        zip.write_all(PRESENTATION_XML.as_bytes()).unwrap();

        zip.add_directory("ppt/_rels", options.clone()).unwrap();
        zip.start_file("ppt/_rels/presentation.xml.rels", options.clone())
            .unwrap();
        zip.write_all(PRESENTATION_RELS.as_bytes()).unwrap();

        zip.add_directory("ppt/slides", options.clone()).unwrap();
        zip.start_file("ppt/slides/slide1.xml", options.clone())
            .unwrap();
        zip.write_all(SLIDE1_PPTX.as_bytes()).unwrap();

        zip.finish().unwrap();
    }

    fs::write(path, buffer.into_inner()).expect("Failed to write PPTX fixture");
}

const CONTENT_TYPES_PPTX: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
    <Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
</Types>"#;

const RELS_PPTX: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"#;

const PRESENTATION_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:sldIdLst>
        <p:sldId id="256" r:id="rId1"/>
    </p:sldIdLst>
</p:presentation>"#;

const PRESENTATION_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
</Relationships>"#;

const SLIDE1_PPTX: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
    <p:cSld>
        <p:spTree>
            <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
            <p:grpSpPr/>
            <p:sp>
                <p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
                <p:spPr/>
                <p:txBody><a:bodyPr/><a:p><a:r><a:t>Test Presentation</a:t></a:r></a:p></p:txBody>
            </p:sp>
            <p:sp>
                <p:nvSpPr><p:cNvPr id="3" name="Content"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
                <p:spPr/>
                <p:txBody><a:bodyPr/><a:p><a:r><a:t>This is a test slide for Momo.</a:t></a:r></a:p></p:txBody>
            </p:sp>
        </p:spTree>
    </p:cSld>
</p:sld>"#;
