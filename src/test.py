import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import model
from sklearn.preprocessing import StandardScaler
import tqdm
import torch
import os


def test(test_loader, input_dim):
    results = []
    models = model.DNN(input_dim)
    models.load_state_dict(torch.load('../src/fadam-batch16-epoch30.pth'))
    models.eval()
    files = os.listdir('../data/test')
    files = sorted(files)
    with torch.no_grad():
        for i, (X_batch,) in enumerate(test_loader):
            print(f'Input batch shape: {X_batch.shape}')
            outputs = models(X_batch)
            print("output")
            print(len(outputs))
            for j, output in (enumerate(outputs)):
                idx = files[i * test_loader.batch_size + j]
                file_name = idx.replace('.ogg', '')
                fake_prob, real_prob = output[0].item(), output[1].item()
                results.append((file_name, fake_prob, real_prob))                # print('file_name: ', file_name)
                # print('fake: ', 1 - output[0])
                # print('real: ', output[0])
    return results


def ensemble_test(test_loader, models):
    results = []
    files = os.listdir('../data/test')
    files = sorted(files)

    with torch.no_grad():
        for i, (X_batch,) in enumerate(test_loader):
            ensemble_outputs = torch.zeros(X_batch.size(0), 2)
            for model in models:
                outputs = model(X_batch)
                ensemble_outputs += outputs
            ensemble_outputs /= len(models)

            for j, output in enumerate(ensemble_outputs):
                idx = files[i * test_loader.batch_size + j]
                file_name = idx.replace('.ogg', '')
                fake_prob, real_prob = output[0].item(), output[1].item()
                results.append((file_name, fake_prob, real_prob))
    return results

def save_result(results, output_csv_path):
    sample_submission = pd.read_csv('../data/sample_submission.csv')
    results_converted = [(id, fake, real) for id, fake, real in results]
    results_df = pd.DataFrame(results_converted, columns=['id', 'fake', 'real'])

    final_submission = sample_submission.drop(columns=['fake', 'real']).merge(results_df, on='id', how='left')

    final_submission.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
