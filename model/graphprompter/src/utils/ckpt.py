import os
import torch


def print_trainable_params(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def _save_checkpoint(model, optimizer, cur_epoch, args, is_best=False):
    """
    Save the checkpoint at the current epoch.
    """

    os.makedirs(args.output_dir, exist_ok=True)

    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    save_obj = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "config": args,
        "epoch": cur_epoch,
    }

    path = f'{args.dataset}_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}'
    save_to = os.path.join(
        args.output_dir,
        path+"_checkpoint_{}.pth".format("best" if is_best else cur_epoch),
    )

    print("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
    torch.save(save_obj, save_to)


def _reload_best_model(model, args):
    """
    Load the best checkpoint for evaluation.
    """
    # if "st" in args.dataset:
    #     path = f'{args.dataset}_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_4.pth'
    #     checkpoint_path = os.path.join(args.output_dir, path)
    # else:
    #     if "cora_semi" in args.dataset:
    #         path = f'cora_semi_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path)
    #     elif "pubmed_semi" in args.dataset:
    #         path = f'pubmed_semi_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path)
    #     elif "products_semi" in args.dataset:
    #         path = f'products_semi_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path)
    #     elif "cora_sup" in args.dataset:
    #         path = f'cora_sup_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_4.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path) 
    #     elif "pubmed_sup" in args.dataset:
    #         path = f'pubmed_sup_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path) 
    #     elif "products_sup" in args.dataset:
    #         path = f'products_sup_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path)
    #     elif "computers_sup" in args.dataset:
    #         path = f'computers_sup_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path) 
    #     elif "sports_sup" in args.dataset:
    #         path = f'sports_sup_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path) 
    #     elif "arxiv_sup" in args.dataset:
    #         path = f'arxiv_sup_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path) 
    #     else:
    #         path = f'{args.dataset}_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    #         checkpoint_path = os.path.join(args.output_dir, path)
    # path = f'{args.dataset}_{args.model_name}_{args.llm_model_name}_{args.gnn_model_name}_seed{args.seed}_checkpoint_best.pth'
    # checkpoint_path = os.path.join(args.output_dir, path)
    checkpoint_path = args.ckpt_path
    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model


def _reload_model(model, checkpoint_path):
    """
    Load the best checkpoint for evaluation.
    """

    print("Loading checkpoint from {}.".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)

    return model
