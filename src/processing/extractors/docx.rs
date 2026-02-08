use super::ExtractedContent;
use crate::error::{MomoError, Result};
use crate::models::DocumentType;

pub struct DocxExtractor;

impl DocxExtractor {
    pub fn extract(bytes: &[u8]) -> Result<ExtractedContent> {
        let docx = docx_rs::read_docx(bytes)
            .map_err(|e| MomoError::Processing(format!("DOCX parse error: {}", e)))?;

        let mut text = String::new();
        let mut title = None;

        // Process document children
        for child in &docx.document.children {
            match child {
                docx_rs::DocumentChild::Paragraph(paragraph) => {
                    let para_text = Self::extract_paragraph(paragraph);
                    if !para_text.trim().is_empty() {
                        // Check if this is a heading and could be the title
                        if title.is_none() {
                            if let Some(style) = &paragraph.property.style {
                                if style.val == "Heading1" || style.val == "Title" {
                                    title = Some(para_text.trim_start_matches("# ").to_string());
                                }
                            }
                        }

                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&para_text);
                    }
                }
                docx_rs::DocumentChild::Table(table) => {
                    let table_text = Self::extract_table(table);
                    if !table_text.is_empty() {
                        if !text.is_empty() {
                            text.push('\n');
                        }
                        text.push_str(&table_text);
                    }
                }
                _ => {}
            }
        }

        let word_count = Self::count_words(&text);

        Ok(ExtractedContent {
            text,
            title,
            doc_type: DocumentType::Docx,
            url: None,
            word_count,
            source_path: None,
        })
    }

    fn extract_paragraph(paragraph: &docx_rs::Paragraph) -> String {
        // Check for heading style
        let heading_prefix = if let Some(style) = &paragraph.property.style {
            if style.val.starts_with("Heading") {
                // Extract heading level from "Heading1", "Heading2", etc.
                if let Some(level_str) = style.val.strip_prefix("Heading") {
                    if let Ok(level) = level_str.parse::<u8>() {
                        if level >= 1 && level <= 6 {
                            "#".repeat(level as usize) + " "
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else if style.val == "Title" {
                "# ".to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        // Check for list formatting
        let list_prefix = if heading_prefix.is_empty() {
            Self::get_list_prefix(paragraph)
        } else {
            String::new()
        };

        // Extract text content from the paragraph
        let mut content = String::new();
        for para_child in &paragraph.children {
            if let docx_rs::ParagraphChild::Run(run) = para_child {
                for run_child in &run.children {
                    if let docx_rs::RunChild::Text(text) = run_child {
                        content.push_str(&text.text);
                    }
                }
            }
        }

        // Combine prefix and content
        if !content.is_empty() {
            if !heading_prefix.is_empty() {
                heading_prefix + &content
            } else if !list_prefix.is_empty() {
                list_prefix + &content
            } else {
                content
            }
        } else {
            String::new()
        }
    }

    fn get_list_prefix(paragraph: &docx_rs::Paragraph) -> String {
        // Check if paragraph has numbering properties
        if let Some(numbering) = &paragraph.property.numbering_property {
            // Check for numbered list (ilvl indicates list level)
            if let Some(ilvl) = &numbering.level {
                let level = ilvl.val as usize;
                let indent = "  ".repeat(level);

                // Check if it's a numbered list by looking at the num_id
                // In Word, different num_ids indicate different list types
                if let Some(num_id) = &numbering.id {
                    // For simplicity, we use alternating patterns to distinguish bullet vs numbered
                    // Even num_ids typically indicate bullet lists, odd indicate numbered lists
                    let num = num_id.id;
                    if num % 2 == 0 {
                        format!("{}- ", indent)
                    } else {
                        format!("{}1. ", indent)
                    }
                } else {
                    format!("{}- ", indent)
                }
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    }

    fn extract_table(table: &docx_rs::Table) -> String {
        let mut rows: Vec<Vec<String>> = Vec::new();

        for table_child in &table.rows {
            let docx_rs::TableChild::TableRow(row) = table_child;
            let mut row_cells: Vec<String> = Vec::new();
            for row_child in &row.cells {
                let docx_rs::TableRowChild::TableCell(cell) = row_child;
                // Extract text from cell
                let mut cell_text = String::new();
                for cell_child in &cell.children {
                    if let docx_rs::TableCellContent::Paragraph(para) = cell_child {
                        let para_text = Self::extract_paragraph_text_only(&para);
                        if !cell_text.is_empty() {
                            cell_text.push(' ');
                        }
                        cell_text.push_str(&para_text);
                    }
                }
                row_cells.push(cell_text.trim().to_string());
            }
            if !row_cells.is_empty() {
                rows.push(row_cells);
            }
        }

        if rows.is_empty() {
            return String::new();
        }

        // Convert to markdown table
        let mut result = String::new();
        let col_count = rows[0].len();

        // Header row
        result.push_str("| ");
        for (i, cell) in rows[0].iter().enumerate() {
            result.push_str(cell);
            if i < col_count - 1 {
                result.push_str(" | ");
            }
        }
        result.push_str(" |\n");

        // Separator
        result.push_str("|");
        for _ in 0..col_count {
            result.push_str("------|");
        }
        result.push('\n');

        // Data rows
        for row in rows.iter().skip(1) {
            result.push_str("| ");
            for (i, cell) in row.iter().enumerate() {
                result.push_str(cell);
                if i < col_count - 1 {
                    result.push_str(" | ");
                }
            }
            result.push_str(" |\n");
        }

        result
    }

    fn extract_paragraph_text_only(paragraph: &docx_rs::Paragraph) -> String {
        let mut content = String::new();
        for para_child in &paragraph.children {
            if let docx_rs::ParagraphChild::Run(run) = para_child {
                for run_child in &run.children {
                    if let docx_rs::RunChild::Text(text) = run_child {
                        content.push_str(&text.text);
                    }
                }
            }
        }
        content
    }

    fn count_words(text: &str) -> i32 {
        text.split_whitespace().count() as i32
    }
}
