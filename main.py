from ultralytics import YOLO
import cv2
import requests
import time
import threading

# ==========================================
# CONFIGURAÇÕES
# ==========================================

# Modelo de IA
MODEL_PATH = "yolo11n.pt"

# Câmera do Redmi
URL_CAMERA = "http://192.168.1.21:8080/video"

# API do site
API_URL = "https://project--bd20e6c8-95d5-4277-bf1f-ebe6b57911fd-dev.lovable.app/api/public/lotacao"

# Capacidade máxima do ônibus
CAPACIDADE_ONIBUS = 40


# ==========================================
# DESEMPENHO
# ==========================================

# Quantidade de análises do YOLO por segundo
FPS_YOLO = 5

# 5 FPS = uma análise a cada 0,2 segundo
INTERVALO_DETECCAO = 1.0 / FPS_YOLO

# Envio para a API a cada 1 segundo
INTERVALO_ENVIO = 1.0

# Tamanho da imagem usada pelo YOLO
TAMANHO_IMAGEM = 640


# ==========================================
# CARREGA O MODELO
# ==========================================

print("Carregando modelo YOLO...")

model = YOLO(MODEL_PATH)

print("Modelo carregado!")


# ==========================================
# ENVIO PARA API
# ==========================================

def enviar_para_api(dados):

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
                f"Erro da API: "
                f"{resposta.status_code} - "
                f"{resposta.text}"
            )

    except requests.exceptions.RequestException as erro:

        print(f"Falha ao enviar dados para a API: {erro}")


# ==========================================
# CONECTA À CÂMERA
# ==========================================

print("Conectando à câmera...")

cap = cv2.VideoCapture(URL_CAMERA)

# Tenta reduzir o buffer da câmera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():

    print("ERRO: não foi possível conectar à câmera.")

    exit()


print("Câmera conectada!")
print("API configurada!")
print(f"YOLO configurado para {FPS_YOLO} FPS.")
print("Pressione ESC para sair.")


# ==========================================
# VARIÁVEIS
# ==========================================

ultima_deteccao = 0
ultimo_envio = 0

pessoas = 0
lotacao = 0.0

status = "LIVRE"

cor_status = (0, 255, 0)

frame_processado = None


# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    # --------------------------------------
    # RECEBE FRAME DA CÂMERA
    # --------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("Não foi possível receber o vídeo.")

        break


    agora = time.monotonic()


    # ======================================
    # YOLO - 5 FPS
    # ======================================

    if agora - ultima_deteccao >= INTERVALO_DETECCAO:

        ultima_deteccao = agora

        # ----------------------------------
        # DETECTA SOMENTE PESSOAS
        # ----------------------------------

        results = model(
            frame,
            classes=[0],
            imgsz=TAMANHO_IMAGEM,
            verbose=False
        )


        # ----------------------------------
        # CONTA PESSOAS
        # ----------------------------------

        pessoas = 0

        for result in results:

            for box in result.boxes:

                pessoas += 1


        # ----------------------------------
        # CALCULA LOTAÇÃO
        # ----------------------------------

        lotacao = (
            pessoas / CAPACIDADE_ONIBUS
        ) * 100


        # ----------------------------------
        # DEFINE STATUS
        # ----------------------------------

        if lotacao <= 50:

            status = "LIVRE"

            cor_status = (0, 255, 0)

        elif lotacao <= 80:

            status = "MODERADO"

            cor_status = (0, 255, 255)

        else:

            status = "LOTADO"

            cor_status = (0, 0, 255)


        # ----------------------------------
        # DESENHA DETECÇÕES
        # ----------------------------------

        frame_processado = results[0].plot()


        # ==================================
        # ENVIA DADOS PARA API
        # ==================================

        if agora - ultimo_envio >= INTERVALO_ENVIO:

            ultimo_envio = agora

            dados = {
                "pessoas": int(pessoas),
                "capacidade": int(CAPACIDADE_ONIBUS),
                "lotacao": round(float(lotacao), 1),
                "status": str(status)
            }

            # Envia em segundo plano para
            # não travar o vídeo
            threading.Thread(
                target=enviar_para_api,
                args=(dados,),
                daemon=True
            ).start()


    # ======================================
    # CASO AINDA NÃO TENHA DETECÇÃO
    # ======================================

    if frame_processado is None:

        frame_processado = frame.copy()


    # ======================================
    # INFORMAÇÕES NA TELA
    # ======================================

    # Pessoas

    cv2.putText(
        frame_processado,
        f"Pessoas: {pessoas}/{CAPACIDADE_ONIBUS}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    # Lotação

    cv2.putText(
        frame_processado,
        f"Lotacao: {lotacao:.1f}%".replace(".", ","),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    # Status

    cv2.putText(
        frame_processado,
        f"Status: {status}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        cor_status,
        3
    )


    # ======================================
    # MOSTRA O VÍDEO
    # ======================================

    cv2.imshow(
        "IA - Lotacao do Onibus",
        frame_processado
    )


    # ======================================
    # ESC PARA SAIR
    # ======================================

    if cv2.waitKey(1) & 0xFF == 27:

        break


# ==========================================
# ENCERRAMENTO
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("Sistema encerrado.")