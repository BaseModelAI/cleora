use cleora::configuration::{Column, Configuration};
use cleora::embedding::calculate_embeddings;
use cleora::persistence::embedding::EmbeddingPersistor;
use cleora::persistence::entity::InMemoryEntityMappingPersistor;
use cleora::pipeline::build_graphs;
use insta::assert_debug_snapshot;
use std::sync::Arc;

/// This test performs work for sample case and saves snapshot file.
/// Snapshot testing takes advantage of deterministic character of Cleora.
/// Any discrepancies between original snapshot results and current ones can be then
/// reviewed along with the code which introduced discrepancy.
///
/// Differing snapshot has to be renamed by removing .new from the name.
/// For more information, please review https://crates.io/crates/insta
///
/// Code executed performs roughly the same work as:
/// ./cleora -i files/samples/edgelist_1.tsv --columns="complex::reflexive::a b complex::c"
/// -d 128 -n 4 --relation-name=R1 -p 0
#[test]
fn test_build_graphs_and_create_embedding() {
    let columns = vec![
        Column {
            name: "a".to_string(),
            complex: true,
            reflexive: true,
            ..Column::default()
        },
        Column {
            name: "b".to_string(),
            ..Column::default()
        },
        Column {
            name: "c".to_string(),
            complex: true,
            ..Column::default()
        },
    ];

    let config = Configuration {
        produce_entity_occurrence_count: true,
        embeddings_dimension: 128,
        max_number_of_iteration: 4,
        prepend_field: false,
        log_every_n: 10000,
        in_memory_embedding_calculation: true,
        input: "files/samples/edgelist_1.tsv".to_string(),
        output_dir: None,
        relation_name: "r1".to_string(),
        columns,
    };
    let in_memory_entity_mapping_persistor = InMemoryEntityMappingPersistor::new();
    let in_memory_entity_mapping_persistor = Arc::new(in_memory_entity_mapping_persistor);

    // build sparse matrices
    let mut sparse_matrices = build_graphs(&config, in_memory_entity_mapping_persistor.clone());
    assert_debug_snapshot!("sparse_matrices", sparse_matrices);

    let mut in_memory_embedding_persistor = InMemoryEmbeddingPersistor::default();

    // calculate embeddings for ONE sparse matrix
    calculate_embeddings(
        &mut sparse_matrices[0],
        config.max_number_of_iteration,
        in_memory_entity_mapping_persistor,
        &mut in_memory_embedding_persistor,
    );

    assert_debug_snapshot!("embeddings", in_memory_embedding_persistor);
}

#[derive(Debug, Default)]
pub struct InMemoryEmbeddingPersistor {
    entity_count: u32,
    dimenstion: u16,
    entities: Vec<InMemoryEntity>,
}

#[derive(Debug)]
pub struct InMemoryEntity {
    entity: String,
    occur_count: u32,
    vector: Vec<f32>,
}

impl EmbeddingPersistor for InMemoryEmbeddingPersistor {
    fn put_metadata(&mut self, entity_count: u32, dimension: u16) {
        self.entity_count = entity_count;
        self.dimenstion = dimension;
    }
    fn put_data(&mut self, entity: String, occur_count: u32, vector: Vec<f32>) {
        self.entities.push(InMemoryEntity {
            entity,
            occur_count,
            vector,
        });
    }
    fn finish(&mut self) {}
}
