import argparse 






def main(args):
    
    pass




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the DGE on dataset")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset string")
    parser.add_argument('--clusters', type=int, default=10,
                        help="Number of clusters for clustering layer.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate for optimizer.")
    parser.add_argument('--epochs', type=int, default=500,
                        help="Number of training epochs.")
    parser.add_argument('--hidden', type=str, default="512,256,128",
                        help="Comma-separated list of hidden dimensions for the encoder.")
    parser.add_argument('--graph_method', type=str, default="pearson", choices=["pearson", "spearman", "NE"],
                        help="Method to compute the graph connectivity from features.")
    parser.add_argument('--pre_lr', type=float, default=0.001,
                        help="Learning rate for pre-training optimizer.")
    parser.add_argument('--c1', type=float, default=1.0,
                        help="Weight for reconstruction loss.")
    parser.add_argument('--c2', type=float, default=0.5,
                        help="Weight for cluster loss.")
    parser.add_argument('--subtype_path', type=str, default=None,
                        help="Path to the subtype file.")

    args = parser.parse_args()
    main(args)
