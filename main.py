from ultralytics import YOLO
import cv2
import requests
import time
import threading

# ==========================================
# CARREGA O MODELO DE IA
# ==========================================

model = YOLO("yolo11n.pt")

# ==========================================
# CÂMERA DO REDMI ATRAVÉS DO TAILSCALE
# ==========================================

URL_CAMERA = "http://192.168.1.33:8080/video"

# ==========================================
# API DO SITE
# ==========================================

API_URL = "https://project--bd20e6c8-95d5-4277-bf1f-ebe6b57911fd-dev.lovable.app/api/public/lotacao"

# Envia os dados no máximo uma vez por segundo
INTERVALO_ENVIO = 1.0
ultimo_envio = 0

# ==========================================
# CAPACIDADE MÁXIMA
# ==========================================

CAPACIDADE_ONIBUS = 40

# ==========================================
# FUNÇÃO PARA ENVIAR DADOS PARA O SITE
# ==========================================

def enviar_para_api(pessoas, capacidade, lotacao, status):

    dados = {
        "pessoas": int(pessoas),
        "capacidade": int(capacidade),
        "lotacao": round(float(lotacao), 1),
        "status": str(status)
    }

    try:
        resposta = requests.post(
            API_URL,
            json=dados,
            timeout=3
        )

        if resposta.ok:
            print(f"Dados enviados: {dados}")
        else:
            print(
                f"Erro da API: {resposta.status_code} "
                f"- {resposta.text}"
            )

    except requests.exceptions.RequestException as erro:
        print(f"Falha ao enviar dados para a API: {erro}")


# ==========================================
# CONECTA À CÂMERA
# ==========================================

cap = cv2.VideoCapture(URL_CAMERA)

if not cap.isOpened():
    print("ERRO: não foi possível conectar à câmera.")
    exit()

print("Câmera conectada!")
print("API configurada!")
print("Pressione ESC para sair.")

# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    # Recebe um frame da câmera
    ret, frame = cap.read()

    if not ret:
        print("Não foi possível receber o vídeo.")
        break

    # ==========================================
    # IA DETECTA SOMENTE PESSOAS
    # Classe 0 = pessoa
    # ==========================================

    results = model(
        frame,
        classes=[0],
        verbose=False
    )

    # Contador de pessoas
    pessoas = 0

    # ==========================================
    # CONTA AS PESSOAS DETECTADAS
    # ==========================================

    for result in results:

        for box in result.boxes:

            # Como estamos detectando somente classe 0,
            # cada box já representa uma pessoa.
            pessoas += 1

    # ==========================================
    # CALCULA A PORCENTAGEM DE LOTAÇÃO
    # ==========================================

    lotacao = (pessoas / CAPACIDADE_ONIBUS) * 100

    # ==========================================
    # DEFINE O STATUS
    # ==========================================

    if lotacao <= 50:

        status = "LIVRE"
        cor_status = (0, 255, 0)       # Verde

    elif lotacao <= 80:

        status = "MODERADO"
        cor_status = (0, 255, 255)     # Amarelo

    else:

        status = "LOTADO"
        cor_status = (0, 0, 255)       # Vermelho

    # ==========================================
    # ENVIA OS DADOS PARA O SITE
    # ==========================================

    agora = time.time()

    if agora - ultimo_envio >= INTERVALO_ENVIO:

        ultimo_envio = agora

        # Usa uma thread para o envio não travar
        # a captura e processamento do vídeo.
        threading.Thread(
            target=enviar_para_api,
            args=(
                pessoas,
                CAPACIDADE_ONIBUS,
                lotacao,
                status
            ),
            daemon=True
        ).start()

    # ==========================================
    # DESENHA SOMENTE AS PESSOAS DETECTADAS
    # ==========================================

    frame = results[0].plot()

    # ==========================================
    # INFORMAÇÕES NA TELA
    # ==========================================

    # Pessoas
    cv2.putText(
        frame,
        f"Pessoas: {pessoas}/{CAPACIDADE_ONIBUS}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    # Lotação
    cv2.putText(
        frame,
        f"Lotacao: {lotacao:.1f}%".replace(".", ","),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    # Status
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        cor_status,
        3
    )

    # ==========================================
    # MOSTRA O VÍDEO
    # ==========================================

    cv2.imshow(
        "IA - Lotacao do Onibus",
        frame
    )

    # ==========================================
    # ESC PARA SAIR
    # ==========================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================
# ENCERRA A CÂMERA E FECHA AS JANELAS
# ==========================================

cap.release()
cv2.destroyAllWindows()