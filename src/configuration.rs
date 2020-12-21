#[derive(Debug)]
pub enum FileType {
    JSON,
    TSV,
}

#[derive(Debug)]
pub enum OutputFormat {
    TextFile,
    Numpy,
}

/// Pipeline configuration
#[derive(Debug)]
pub struct Configuration {
    /// Produce or not entity counter to the output file
    pub produce_entity_occurrence_count: bool,

    /// Dimension of the embedding
    pub embeddings_dimension: u16,

    /// Maximum number of iteration for training
    pub max_number_of_iteration: u8,

    /// Prepend field name to entity in the output file. It differentiates entities with the same
    /// name from different columns
    pub prepend_field: bool,

    /// After how many lines we log the progress
    pub log_every_n: u32,

    /// Calculate embeddings in memory or with memory-mapped files. If we don't have enough
    /// RAM we can support training with mmap files
    pub in_memory_embedding_calculation: bool,

    /// Path to the input file
    pub input: String,

    /// Type of the input file
    pub file_type: FileType,

    /// Output directory for files with embeddings
    pub output_dir: Option<String>,

    /// Output format
    pub output_format: OutputFormat,

    /// Name of the relation, for output filename generation
    pub relation_name: String,

    /// Columns configuration
    pub columns: Vec<Column>,
}

/// Column configuration
#[derive(Debug, Default)]
pub struct Column {
    /// Name, header of the column
    pub name: String,

    /// The field is virtual - it is considered during embedding process, no entity is written for the column
    pub transient: bool,

    /// The field is composite, containing multiple entity identifiers separated by space
    pub complex: bool,

    /// The field is reflexive, which means that it interacts with itself, additional output file is written for every such field
    pub reflexive: bool,

    /// The field is ignored, no output file is written for the field
    pub ignored: bool,
}

impl Configuration {
    /// Create default configuration with specified input file path and columns.
    pub fn default(input: String, columns: Vec<Column>) -> Configuration {
        Configuration {
            produce_entity_occurrence_count: true,
            embeddings_dimension: 128,
            max_number_of_iteration: 4,
            prepend_field: true,
            log_every_n: 1000,
            in_memory_embedding_calculation: true,
            file_type: FileType::TSV,
            input,
            output_dir: None,
            output_format: OutputFormat::TextFile,
            relation_name: String::from("emb"),
            columns,
        }
    }

    /// Filter out ignored columns. Entities from such columns are omitted.
    pub fn not_ignored_columns(&self) -> Vec<&Column> {
        self.columns.iter().filter(|&c| !c.ignored).collect()
    }
}

/// Extract columns config based on raw strings.
pub fn extract_fields(cols: Vec<&str>) -> Result<Vec<Column>, String> {
    let mut columns: Vec<Column> = Vec::new();

    for col in cols {
        let parts: Vec<&str> = col.split("::").collect();

        let column_name: &str;
        let mut transient = false;
        let mut complex = false;
        let mut reflexive = false;
        let mut ignored = false;

        let parts_len = parts.len();
        if parts_len > 1 {
            column_name = *parts.last().unwrap();
            let column_name_idx = parts_len - 1;
            for &part in &parts[..column_name_idx] {
                if part.eq_ignore_ascii_case("transient") {
                    transient = true;
                } else if part.eq_ignore_ascii_case("complex") {
                    complex = true;
                } else if part.eq_ignore_ascii_case("reflexive") {
                    reflexive = true;
                } else if part.eq_ignore_ascii_case("ignore") {
                    ignored = true;
                } else {
                    let message = format!("Unrecognized column field modifier: {}", part);
                    return Err(message);
                }
            }
        } else {
            column_name = col;
        }
        let column = Column {
            name: column_name.to_string(),
            transient,
            complex,
            reflexive,
            ignored,
        };
        columns.push(column);
    }
    Ok(columns)
}

/// Validate column modifiers.
pub fn validate_fields(cols: Vec<Column>) -> Result<Vec<Column>, String> {
    for col in &cols {
        // transient::reflexive - this would generate no output
        // transient::reflexive::complex - this would generate no output
        if col.reflexive && col.transient {
            let message = format!("A field cannot be REFLEXIVE and simultaneously TRANSIENT. It does not make sense: {}", col.name);
            return Err(message);
        }
        if col.reflexive && !col.complex {
            let message = format!(
                "A field cannot be REFLEXIVE but NOT COMPLEX. It does not make sense: {}",
                col.name
            );
            return Err(message);
        }
    }
    Ok(cols)
}
