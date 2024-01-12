

def get_dataset(args, mode, **kwargs):
    
    if args.dimension == '2d':
        if args.dataset == 'acdc':
            from .dim2.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)
        if args.dataset == 'biobank':
            from .dim2.dataset_biobank import CMRDataset
            return CMRDataset(args,mode=mode,k_fold=args.k_fold,k=kwargs['fold_idx'],seed=args.seed)
        if args.dataset == 'biobank_gen':
            from .dim2.dataset_biobank_gen import CMRDataset
            return CMRDataset(args,mode=mode,k_fold=args.k_fold,k=kwargs['fold_idx'],seed=args.seed)
    else:
        if args.dataset == 'acdc':
            from .dim3.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)
        elif args.dataset == 'lits':
            from .dim3.dataset_lits import LiverDataset

            return LiverDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'bcv':
            from .dim3.dataset_bcv import BCVDataset

            return BCVDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)
        elif args.dataset == "acdc_fft":
            from .dim3.dataset_acdc_fft import CMRDataset
            return CMRDataset(args,mode=mode, k_fold=args.k_fold,k=kwargs['fold_idx'])


