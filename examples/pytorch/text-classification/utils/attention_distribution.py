import torch
import tqdm
from run_glue import main as rg_main


def main():
    trainer, training_args, data_args, model_args = rg_main(outside_usage=True)
    model = trainer.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    entropy_sum = torch.zeros((model.config.num_hidden_layers,), dtype=torch.float64, device=device)
    count = 0
    i = 0
    for b in tqdm.tqdm(trainer.get_train_dataloader()):
        b = trainer._prepare_inputs(b)
        outputs = model(output_attentions=True, **b)
        attention_mask = b['attention_mask']
        for l in range(len(outputs.attentions)):
            probs = outputs.attentions[l]
            probs = probs[attention_mask.unsqueeze(-1).expand(-1, -1, probs.shape[1]).transpose(1, 2).bool()]
            mean_entropy = (-probs*probs.log())[probs.bool()].sum() / probs.shape[0]
            entropy_sum[l] += mean_entropy
        count += 1
        if i % 100 == 0:
            print(entropy_sum / count)
        i += 1
    entropy_sum /= count
    print(entropy_sum)





if __name__ == '__main__':
    with torch.no_grad():
        main()
