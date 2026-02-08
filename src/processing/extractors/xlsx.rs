use super::ExtractedContent;
use crate::error::{MomoError, Result};
use crate::models::DocumentType;
use calamine::{open_workbook_auto_from_rs, Data, Reader};
use std::io::Cursor;

pub struct XlsxExtractor;

impl XlsxExtractor {
    const MAX_ROWS: usize = 100_000;

    pub fn extract(bytes: &[u8]) -> Result<ExtractedContent> {
        let cursor = Cursor::new(bytes);
        let mut workbook = open_workbook_auto_from_rs(cursor)
            .map_err(|e| MomoError::Processing(format!("XLSX parse error: {}", e)))?;

        let mut text = String::new();

        for (name, range) in workbook.worksheets() {
            let (rows, cols) = range.get_size();

            // Guard against pathological files with excessive rows
            let row_limit = rows.min(Self::MAX_ROWS);

            text.push_str(&format!("## Sheet: {}\n\n", name));

            if rows == 0 || cols == 0 {
                // Empty sheet - just add the header and continue
                text.push_str("*(empty sheet)*\n\n");
                continue;
            }

            // Convert range to markdown table
            let mut table_rows: Vec<Vec<String>> = Vec::new();

            for row_idx in 0..row_limit {
                let mut row_cells: Vec<String> = Vec::new();
                for col_idx in 0..cols {
                    let cell_value = range.get_value((row_idx as u32, col_idx as u32));
                    let cell_text = Self::format_cell_value(cell_value);
                    row_cells.push(cell_text);
                }
                table_rows.push(row_cells);
            }

            // Generate markdown table
            if !table_rows.is_empty() {
                // Header row
                text.push_str("| ");
                for (i, cell) in table_rows[0].iter().enumerate() {
                    text.push_str(cell);
                    if i < cols - 1 {
                        text.push_str(" | ");
                    }
                }
                text.push_str(" |\n");

                // Separator
                text.push_str("|");
                for _ in 0..cols {
                    text.push_str("------|");
                }
                text.push('\n');

                // Data rows
                for row in table_rows.iter().skip(1) {
                    text.push_str("| ");
                    for (i, cell) in row.iter().enumerate() {
                        text.push_str(cell);
                        if i < cols - 1 {
                            text.push_str(" | ");
                        }
                    }
                    text.push_str(" |\n");
                }
            }

            if rows > Self::MAX_ROWS {
                text.push_str(&format!(
                    "\n*... truncated (showing {} of {} rows)*\n",
                    Self::MAX_ROWS,
                    rows
                ));
            }

            text.push('\n');
        }

        let word_count = Self::count_words(&text);

        Ok(ExtractedContent {
            text,
            title: None,
            doc_type: DocumentType::Xlsx,
            url: None,
            word_count,
            source_path: None,
        })
    }

    fn format_cell_value(cell: Option<&Data>) -> String {
        match cell {
            Some(Data::String(s)) => s.clone(),
            Some(Data::Int(i)) => i.to_string(),
            Some(Data::Float(f)) => {
                // Format float nicely - remove trailing zeros
                let s = format!("{}", f);
                if s.contains('.') {
                    s.trim_end_matches('0').trim_end_matches('.').to_string()
                } else {
                    s
                }
            }
            Some(Data::Bool(b)) => b.to_string(),
            Some(Data::DateTime(dt)) => {
                // Format as ISO8601
                dt.to_string()
            }
            Some(Data::DateTimeIso(dt)) => dt.to_string(),
            Some(Data::DurationIso(d)) => d.to_string(),
            Some(Data::Empty) | None => String::new(),
            _ => String::new(),
        }
    }

    fn count_words(text: &str) -> i32 {
        text.split_whitespace().count() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_cell_value() {
        assert_eq!(
            XlsxExtractor::format_cell_value(Some(&Data::String("hello".to_string()))),
            "hello"
        );
        assert_eq!(XlsxExtractor::format_cell_value(Some(&Data::Int(42))), "42");
        assert_eq!(
            XlsxExtractor::format_cell_value(Some(&Data::Float(3.14))),
            "3.14"
        );
        assert_eq!(
            XlsxExtractor::format_cell_value(Some(&Data::Bool(true))),
            "true"
        );
        assert_eq!(XlsxExtractor::format_cell_value(Some(&Data::Empty)), "");
        assert_eq!(XlsxExtractor::format_cell_value(None), "");
    }
}
