import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_ece(confidences, accuracies, n_bins=10):
    """
    Calcula o Expected Calibration Error (ECE) de acordo com Guo et al. (2017).

    Args:
        confidences (np.array): Confianças da IA (0-100).
        accuracies (np.array): Notas Humanas / Acurácia real (0-100).
        n_bins (int): Número de agrupamentos (padrão 10).

    Returns:
        ece (float): Valor do ECE Global.
        bin_confs (list): Confiança média por bin.
        bin_accs (list): Acurácia média por bin.
    """
    bin_boundaries = np.linspace(0, 100, n_bins + 1)
    ece = 0
    n = len(confidences)

    bin_confs = []
    bin_accs = []

    for i in range(n_bins):
        # Define os limites do bin atual
        lower = bin_boundaries[i]
        upper = bin_boundaries[i+1]

        # Filtra as amostras que caem neste bin de confiança
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        bin_size = np.sum(mask)

        if bin_size > 0:
            # Cálculo da acurácia média e confiança média do bin
            avg_acc = np.mean(accuracies[mask])
            avg_conf = np.mean(confidences[mask])

            # Acumula o erro ponderado para o ECE Global
            ece += (bin_size / n) * np.abs(avg_acc - avg_conf)

            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
        else:
            # Mantém valores nulos para bins vazios para não quebrar o gráfico
            bin_confs.append(np.nan)
            bin_accs.append(np.nan)

    return ece, bin_confs, bin_accs

def plot_reliability_diagram(res_fiel, res_hig, ece_fiel, ece_hig):
    """
    Gera o Reliability Diagram comparativo.
    """
    plt.figure(figsize=(10, 8))

    # Linha de Calibração Perfeita (Diagonal)
    plt.plot([0, 100], [0, 100], '--', color='gray', label='Calibração Perfeita', alpha=0.7)

    # Plot da Versão Fiel (Com hesitações)
    # Filtramos NaNs para o plot conectar apenas bins existentes
    mask_fiel = ~np.isnan(res_fiel['accs'])
    plt.plot(np.array(res_fiel['confs'])[mask_fiel],
             np.array(res_fiel['accs'])[mask_fiel],
             'o-', color='#1f77b4', label=f'Versão Fiel (ECE: {ece_fiel:.2f})', linewidth=2)

    # Plot da Versão Higienizada (Sem hesitações)
    mask_hig = ~np.isnan(res_hig['accs'])
    plt.plot(np.array(res_hig['confs'])[mask_hig],
             np.array(res_hig['accs'])[mask_hig],
             'o-', color='#d62728', label=f'Versão Higienizada (ECE: {ece_hig:.2f})', linewidth=2)

    # Formatação do Gráfico
    plt.title('Reliability Diagram: Impacto da Hesitação na Calibração do LLM', fontsize=14, pad=20)
    plt.xlabel('Confiança Prevista pela IA (Logit Score)', fontsize=12)
    plt.ylabel('Certeza Real (Nota Humana)', fontsize=12)
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)

    # Adicionando uma zona de Overconfidence vs Underconfidence
    plt.text(70, 20, "Overconfidence\n(Alucinação de Certeza)", color='darkred', alpha=0.5, fontsize=10, ha='center')
    plt.text(20, 80, "Underconfidence", color='darkblue', alpha=0.5, fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig('stil_reliability_diagram_comparativo.png', dpi=300)
    plt.show()

def main():
    # 1. Carregamento dos dados
    path = '/content/resultados_calibrados_token (1).csv'
    try:
        df = pd.read_csv(path)
        print(f"Dados carregados com sucesso. N = {len(df)}")
    except FileNotFoundError:
        print(f"Erro: O arquivo {path} não foi encontrado.")
        return

    # 2. Cálculo do ECE para ambas as condições
    # Fiel (com hesitações)
    ece_fiel, confs_fiel, accs_fiel = calculate_ece(
        df['CONFIANCA_IA_FIEL'].values,
        df['NOTA_HUMANA'].values
    )

    # Higienizada (limpa)
    ece_hig, confs_hig, accs_hig = calculate_ece(
        df['CONFIANCA_IA_HIGIENIZADA'].values,
        df['NOTA_HUMANA'].values
    )

    # 3. Exibição dos Resultados no Terminal
    print("\n" + "="*40)
    print("RESULTADOS DE CALIBRAÇÃO (GLOBAL ECE)")
    print("="*40)
    print(f"ECE (Fila Fiel - Com Hesitação):    {ece_fiel:.4f}")
    print(f"ECE (Fila Higienizada - Limpa):     {ece_hig:.4f}")
    print(f"Diferença de Erro (Delta ECE):      {np.abs(ece_fiel - ece_hig):.4f}")
    print("="*40)

    # 4. Geração do Gráfico
    res_fiel = {'confs': confs_fiel, 'accs': accs_fiel}
    res_hig = {'confs': confs_hig, 'accs': accs_hig}

    plot_reliability_diagram(res_fiel, res_hig, ece_fiel, ece_hig)

if __name__ == "__main__":
    main()
