
#!/usr/bin/env bash

set -uo pipefail  # NÃO uso -e pra não abortar tudo se um repo falhar

echo "========================================"
echo "Pré-aquecimento de imagens Docker"
echo "Iniciado em: $(date)"
echo "========================================"

# Lista de repos a pré-aquecer (cobre todos da amostra)
REPOS=(django sympy sphinx-doc astropy scikit-learn)

# Disco antes
echo ""
echo "Espaço em disco ANTES:"
df -h / | tail -1
echo ""

for repo in "${REPOS[@]}"; do
    echo "========================================"
    echo "Pré-aquecendo: $repo"
    echo "Horário: $(date)"
    echo "========================================"
    
    # Pega primeiro instance_id desse repo nos selecionados
    INSTANCE_ID=$(cat data/selected_instances.json | jq -r '.all_ids[]' | grep "^${repo}__" | head -1)
    
    if [ -z "$INSTANCE_ID" ]; then
        echo "AVISO: Nenhum instance_id encontrado pra $repo, pulando."
        continue
    fi
    
    echo "Usando instance: $INSTANCE_ID"
    
    # Roda o harness com cache_level=instance pra preservar imagens
    python -m swebench.harness.run_evaluation \
        --predictions_path gold \
        --max_workers 1 \
        --instance_ids "$INSTANCE_ID" \
        --run_id "prewarm-${repo}" \
        --dataset_name princeton-nlp/SWE-bench_Verified \
        --cache_level instance
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Pré-aquecimento de $repo: SUCESSO"
    else
        echo "Pré-aquecimento de $repo: FALHOU (exit code $EXIT_CODE) - continuando com próximo"
    fi
    
    echo ""
done

echo "========================================"
echo "Pré-aquecimento finalizado em: $(date)"
echo "========================================"
echo ""

echo "Imagens cacheadas:"
sudo docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -E "sweb|REPOSITORY"
echo ""

echo "Uso total do Docker:"
sudo docker system df
echo ""

echo "Espaço em disco DEPOIS:"
df -h / | tail -1
