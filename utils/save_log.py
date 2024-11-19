from utils.metric import compute_scale_and_shift

def save_log(writer, inputs, outputs, losses, args):
    """
    :type log_path: str
    :type output: Dict[str, torch.tensor]
    :type losses: Dict[str, torch.tensor]
    """
    for key, val in outputs.items():
        outputs[key] = val.detach()

    equi_batch = inputs["gt_depth"].shape[0]
    pred_equi_batch = outputs["pred_depth"].shape[0]
    for i in range(pred_equi_batch):
        if i < equi_batch:
            outputs["pred_depth"][i][inputs["val_mask"][i]] -= outputs["pred_depth"][i][inputs["val_mask"][i]].min()
            outputs["pred_depth"][i][inputs["val_mask"][i]] /= outputs["pred_depth"][i][inputs["val_mask"][i]].max()
            outputs["pred_depth"][i] *= inputs["val_mask"][i]

            inputs["gt_depth"][i][inputs["val_mask"][i]] -= inputs["gt_depth"][i][inputs["val_mask"][i]].min()
            inputs["gt_depth"][i][inputs["val_mask"][i]] /= inputs["gt_depth"][i][inputs["val_mask"][i]].max()
            inputs["gt_depth"][i] *= inputs["val_mask"][i]
        else:
            # Equi pred
            outputs["pred_depth"][i] -= outputs["pred_depth"][i].min()
            outputs["pred_depth"][i] /= outputs["pred_depth"][i].max()

    if "pseudo_depth" in inputs:
        mask = inputs["val_mask"]
        mask_cube = inputs["pseudo_mask"]
        cube_batch = inputs["pseudo_depth"].shape[0]
        for i in range(cube_batch):
            # CUBE pred
            outputs["pred_depth_cube"][i] -= outputs["pred_depth_cube"][i].min()
            outputs["pred_depth_cube"][i] /= outputs["pred_depth_cube"][i].max()

            inputs["pseudo_depth"][i] -= inputs["pseudo_depth"][i].min()
            inputs["pseudo_depth"][i] /= inputs["pseudo_depth"][i].max()


    for l, v in losses.items():
        writer.add_scalar(f"{l}", v, args.cur_epoch)

    equi_batch = inputs["gt_depth"].shape[0]
    for j in range(equi_batch):  # write a maxmimum of four images
        # label
        writer.add_image(f"rgb/{j}", inputs["rgb"][j].data, args.cur_step)
        writer.add_image(f"gt_depth/{j}",
                        inputs["gt_depth"][j].data,
                        args.cur_step)
        writer.add_image(f"pred_depth/{j}",
                        outputs["pred_depth"][j].data,
                        args.cur_step)
        # unlabel
        if "pseudo_depth" in inputs and equi_batch + j < len(inputs["rgb"]):
            writer.add_image(f"rgb/{equi_batch + j}", inputs["rgb"][equi_batch + j].data, args.cur_step)
            writer.add_image(f"pred_depth/{equi_batch + j}",
                            outputs["pred_depth"][equi_batch + j].data,
                            args.cur_step)
    if "pseudo_depth" in inputs:
        cube_batch = inputs["pseudo_depth"].shape[0]
        for j in range(cube_batch):
            writer.add_image(f"pred_depth_cube/{j}",
                            outputs["pred_depth_cube"][j].data,
                            args.cur_step)
            writer.add_image(f"pseudo_depth/{j}",
                            inputs["pseudo_depth"][j].data,
                            args.cur_step)
