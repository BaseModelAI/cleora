#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;
    use ndarray;
    use ndarray::{Array, Array2, ArrayBase, Dim, Ix, OwnedRepr};
    use ndarray_rand::rand::rngs::StdRng;
    use ndarray_rand::rand::{RngCore, SeedableRng};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use cleora::embedding::{MarkovType, NdArrayMatrix};
    use cleora::sparse_matrix::SparseMatrix;

    fn round(arr: Array2<f32>) -> Array2<i32> {
        arr.map(|v| (v * 1000.) as i32)
    }

    #[test]
    fn test_markov_left_01() {
        let (graph, embeddings) = create_graph_embeddings_complex_reflexive();
        let embedding_out = NdArrayMatrix::multiply(&graph, embeddings.view(), MarkovType::Left, 8);
        let embedding_out = round(embedding_out);
        assert_debug_snapshot!(embedding_out);
    }

    #[test]
    fn test_markov_left_02() {
        let (graph, embeddings) = create_graph_embeddings_complex_complex();
        let embedding_out = NdArrayMatrix::multiply(&graph, embeddings.view(), MarkovType::Left, 8);
        let embedding_out = round(embedding_out);
        assert_debug_snapshot!(embedding_out);
    }

    #[test]
    fn test_markov_sym_01() {
        let (graph, embeddings) = create_graph_embeddings_complex_reflexive();
        let embedding_out =
            NdArrayMatrix::multiply(&graph, embeddings.view(), MarkovType::Symmetric, 8);
        let embedding_out = round(embedding_out);
        assert_debug_snapshot!(embedding_out);
    }

    #[test]
    fn test_markov_sym_02() {
        let (graph, embeddings) = create_graph_embeddings_complex_complex();
        let embedding_out =
            NdArrayMatrix::multiply(&graph, embeddings.view(), MarkovType::Symmetric, 8);
        let embedding_out = round(embedding_out);
        assert_debug_snapshot!(embedding_out);
    }

    fn create_graph_embeddings_complex_complex(
    ) -> (SparseMatrix, ArrayBase<OwnedRepr<f32>, Dim<[Ix; 2]>>) {
        let num_embeddings: usize = 100;
        let mut rng: StdRng = SeedableRng::seed_from_u64(21_37);

        let mut edges: Vec<_> = vec![];
        for _ in 0..1000 {
            let col_1_node_1 = rng.next_u32() % (num_embeddings as u32);
            let col_1_node_2 = rng.next_u32() % (num_embeddings as u32);

            let col_2_node_1 = rng.next_u32() % (num_embeddings as u32);
            let col_2_node_2 = rng.next_u32() % (num_embeddings as u32);

            edges.push(format!(
                "{} {}\t{} {}",
                col_1_node_1, col_1_node_2, col_2_node_1, col_2_node_2
            ))
        }
        let edges_ref: Vec<&str> = edges.iter().map(|s| s.as_ref()).collect();
        let graph = SparseMatrix::from_rust_iterator(
            "complex::entity_a complex::entity_b",
            16,
            edges_ref.into_iter(),
            None,
        )
        .unwrap();

        let feature_dim: usize = 32;

        let embeddings = Array::random_using(
            (num_embeddings, feature_dim),
            Uniform::new(0., 10.),
            &mut rng,
        );
        (graph, embeddings)
    }

    fn create_graph_embeddings_complex_reflexive(
    ) -> (SparseMatrix, ArrayBase<OwnedRepr<f32>, Dim<[Ix; 2]>>) {
        let num_embeddings: usize = 100;
        let mut rng: StdRng = SeedableRng::seed_from_u64(21_37);

        let mut edges: Vec<_> = vec![];
        for _ in 0..1000 {
            let node_a = rng.next_u32() % (num_embeddings as u32);
            let node_b = rng.next_u32() % (num_embeddings as u32);
            edges.push(format!("{} {}", node_a, node_b))
        }
        let edges_ref: Vec<&str> = edges.iter().map(|s| s.as_ref()).collect();
        let graph = SparseMatrix::from_rust_iterator(
            "reflexive::complex::entity_id",
            16,
            edges_ref.into_iter(),
            None,
        )
        .unwrap();

        let feature_dim: usize = 32;

        let embeddings = Array::random_using(
            (num_embeddings, feature_dim),
            Uniform::new(0., 10.),
            &mut rng,
        );
        (graph, embeddings)
    }
}
