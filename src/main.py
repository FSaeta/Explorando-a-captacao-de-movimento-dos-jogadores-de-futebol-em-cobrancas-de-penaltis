import cv2
import datetime
import imutils
import numpy as np
import math

import os
import os.path as path

from menu_utils import R, Y, B, C, G
import menu_utils as F_SYS
import menu


# Paths
EXEC_PATH = os.getcwd()
MAIN_PATH = path.dirname(path.realpath(__file__))

if EXEC_PATH != MAIN_PATH:
    # Altera diretório de execução para o diretório do arquivo main
    os.chdir(MAIN_PATH)

OUTPUT_PATH_VIDEOS = path.join(MAIN_PATH, 'media', 'output_media')

SRC_PATH_RAW_VIDEOS = path.join(MAIN_PATH, 'media', 'raw_videos')
SRC_PATH_RAW_VIDEOS2 = path.join(MAIN_PATH, 'media', 'raw_videos2')

# Algoritmo de detecção de poses
PROTO_PATH = path.join(MAIN_PATH, 'extra_files', 'MobileNetSSD_deploy.prototxt')
MODEL_PATH = path.join(MAIN_PATH, 'extra_files', 'MobileNetSSD_deploy.caffemodel')

caffe_model = cv2.dnn.readNetFromCaffe(prototxt=PROTO_PATH, caffeModel=MODEL_PATH)

# Only enable it if you are using OpenVino environment
# caffe_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# caffe_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# BGS_TYPES = ['GMG', 'MOG2', 'MOG', 'KNN', 'CNT']      # FIXME
BGS_TYPES = ['MOG2', 'KNN', 'median']


def get_video_cap_info(cap):

    qtd_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps =  cap.get(cv2.CAP_PROP_FPS)
    tempo = round(qtd_frames / fps, 3)
    width = int(cap.get(3))
    height = int(cap.get(4))

    msg = f"""{G} ------- Informações do Vídeo -------

{Y} FPS: {G}{fps}
{Y} Qtd. Frames: {G}{qtd_frames}
{Y} Tempo: {G}{tempo} s
{Y} Dimensões {G} ({width} X {height}) px
----------------------------------{B}"""
    print(msg)

    infos = {
        'fps': fps,
        'frames': qtd_frames,
        'tempo': tempo,
        'dimensoes': (width, height)
    }
    return infos


def get_median_frame(cap, fCount):
    """ Função para obter o frame correspondente ao plano de fundo
    do vídeo que está sendo tratado. O frame obtido é convertido em
    tons de cinza.

    Args:
        cap (VideoCapturer): Objeto do opencv de captura de vídeo
        fCount (float): quantidade de frames no vídeo

    Returns:
        matrix: frame do vídeo em formato de matriz
    """
    # Obtém 25 números uniformemente aleatórios
    random_frames = np.random.uniform(size=25)
    
    # Obtém os números dos 25 frames que serão utilizados 
    # para extrair o plano de fundo
    frames_ids = fCount * random_frames

    frames = []
    for fid in frames_ids:
        # cv2.CAP_PROP_POS_FRAMES é a propriedade para acessar um frame específico
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        # Faz a leitura do frame que o cap foi setado
        has_frame, frame = cap.read()
        frames.append(frame)

    # Obtém a mediana em tons de cinza do fundo e salva a imagem
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    medianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    save_path = path.join(EXEC_PATH, 'media_src', 'process', 'background_sub.jpg')
    cv2.imwrite(save_path, medianFrame)

    # Seta o capturador para o primeiro frame novamente
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return medianFrame


def get_kernel(ttype):
    """ retorna o melhor kernel para o tipo de operação de melhoria de imagens.

    Args:
        ttype (str): método de melhoria de imagem

    Returns:
        matrix: matrix utilizada como kernel
    """
    kernel = False
    if ttype == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    elif ttype in ['opening', 'closing']:
        kernel = np.ones((3,3), np.uint8)
    return kernel

def get_filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel(filter), iterations=2)
    elif filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, get_kernel(filter), iterations=2)
    elif filter == 'dilation':
        return cv2.dilate(img, get_kernel(filter), iterations=2)
    elif filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, get_kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, get_kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, get_kernel('dilation'), iterations=2)
        return dilation

def get_BGS_subtractor(bgs_type, shadows=True):
    """ Retorna o objeto que será utilizado para a optimização da imagem

    Args:
        bgs_type (str): tipo de optimização utilizada
        shadows (bool): capturar sombras dos objetos. Defaults to True.

    Returns:
        Objeto de optimização ou False
    """
    bgs_type = bgs_type.upper()

    if bgs_type == 'GMG':  # FIXME
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, 
                                                        decisionThreshold=0.8)
    elif bgs_type == 'MOG':  # FIXME
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200, 
                                                        nmixtures=5, 
                                                        backgroundRatio=0.7, 
                                                        noiseSigma=0)
    elif bgs_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history=500, 
                                                  detectShadows=shadows, 
                                                  varThreshold=100)
    elif bgs_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, 
                                                 detectShadows=shadows, 
                                                 dist2Threshold=500)
    elif bgs_type == 'CNT':  # FIXME
        return cv2.bgsegm.createBackgroundSubtractorCNT(useHistory=True, 
                                                        minPixelStability=155,
                                                        maxPixelStability=15*60,
                                                        isParallel=True)

    print(f'{R}Não foi encontrada nenhuma optmização com o nome "{bgs_type}"{B}')
    return False

def escolher_pre_processamento():
    print("Escolha um algoritmo de pré processamento, ou pressione enter para continuar \n"
         f"sem pré processamento. Algoritmos disponíveis: \n {Y} {BGS_TYPES} {B}")
    while True:
        pre_process = input(f'{G}Digite o nome do algoritmo de pré processamento: {B}')
        if pre_process:
            if pre_process in BGS_TYPES:
                print(f'{C}Continuando com o algoritmo {pre_process} ... {B}')
                return pre_process
            print(f'{R}Algoritmo de nome "{pre_process}" inválido !{B}')
        else:
            return pre_process

def pre_process_video(frame, subtractor='GMG', method='combine', resize=0.0, background=False):
    """Método para obter frame pré-processado.

    Args:
        frame: frame utilizado para pré-processar
        subtractor (str): algoritmo de subtração.
        method (str): método utilizado para melhoria de imagem.
        resize (float): fator de redimensionamento da imagem.
        background (bool, frame): Frame de plano de fundo.

    Returns:
        frame: frame pré-processado
    """
    if resize:
        frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)

    if subtractor == 'median':
        # Converte o frame para tons de cinza
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Subtrai a diferença absoluta entre o frame atual e o plano de fundo
        difFrame = cv2.absdiff(frame, background)
        # Remove ruídos mantendo a imagem extraida binarizada em branco (255) ou preto (0)
        th, difFrame = cv2.threshold(difFrame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        filtered = get_filter(difFrame, method)
        res = cv2.bitwise_and(difFrame, difFrame, mask=filtered)

    else:
        bg_mask = get_BGS_subtractor(subtractor).apply(frame)
        filter_mask = get_filter(bg_mask, method)
        res = cv2.bitwise_and(frame, frame, mask=filter_mask)
    
    return res

pontos_bola = {}

def update_pontos_bola(x, y, raio):
    global pontos_bola
    updated = False
    if pontos_bola:
        for coord in pontos_bola.keys():

            if ((x <= coord[0] and x > coord[0] - coord[2]) or (x >= coord[0] and x < coord[0] + coord[2])) and \
                ((y <= coord[1] and y > coord[1] - coord[2]) or (y >= coord[1] and y < coord[1] + coord[2])):
                pontos_bola[coord] += 1
                updated = True
                break
    if not updated:
        pontos_bola.update({(x, y, raio): 1})

    return pontos_bola.copy()

def get_max_point():
    global pontos_bola
    maior = ()
    for ponto, qtd in pontos_bola.items():
        if not maior:
            maior = (ponto, qtd)
        else:
            if qtd > maior[1]:
                maior = (ponto,qtd)

    return maior and maior[0]

def process_video(compare=False, resize=1.0):
    global W
    video_path = F_SYS.pedir_nome_video()
    if not video_path:
        print(f"{C} Não foi possível encontrar o vídeo, voltando para o Menu ...{B}")
    else:
        print(f'{C}Processando o vídeo "{video_path}"{B}')

        # pre_process = escolher_pre_processamento()
        pre_process = ''
        resize = 1

        # Capturando o vídeo
        cap = cv2.VideoCapture(video_path)
        video_infos = get_video_cap_info(cap)

        if pre_process == 'median':
            backgroundFrame = get_median_frame(cap, video_infos['frames'])
            backgroundFrame = cv2.resize(backgroundFrame, (0, 0), fx=resize, fy=resize)

        fps_start_time = datetime.datetime.now()
        fps = 0
        total_frames = 0

        while True:
            has_frame, frame = cap.read()

            if not has_frame:
                print(f"{C}Terminando de processar o vídeo ...{B}")
                break

            if pre_process:
                frame = pre_process_video(frame, subtractor=pre_process, resize=resize, background=backgroundFrame)

            # Para identificar a bola, utilizamos a técnica de blur na imagem para
            # melhorar a detecção de círculos, reduzindo a quantidade de círculos
            # falso positivos encontrados

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurFrame = cv2.GaussianBlur(grayFrame, (11,11), 0)

            # Com a imagem com o blur aplicado, o método HoughCircles do OpenCV é utilizado
            # para encontrar círculos na imagem
            circles = cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1.2, 1000000,
                              param1=31, param2=21, minRadius=8, maxRadius=12)
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cx, cy, raio = i[0], i[1], i[2]
                    pontos_bola = update_pontos_bola(cx, cy, raio)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ponto_circulo = get_max_point() or (0, 0, 0)

        save_name = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.avi'
        writer = cv2.VideoWriter(
            path.join(OUTPUT_PATH_VIDEOS, save_name),
            cv2.VideoWriter_fourcc(*'MJPG'),
            video_infos['fps'],
            video_infos['dimensoes'],
            0
        )
        #writer = cv2.VideoWriter('teste.avi', -1, video_infos['fps'], video_infos['dimensoes'], 0)

        while True:
            has_frame, frame = cap.read()

            if not has_frame:
                print(f"{C}Terminando de processar o vídeo ...{B}")
                break

            frame = cv2.resize(frame, (0, 0), fx=resize, fy=resize)
            total_frames = total_frames + 1
            H, W = frame.shape[:2]

            # Person_detection
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            caffe_model.setInput(blob)
            person_detections = caffe_model.forward()

            person_points = []
            for i in np.arange(0, person_detections.shape[2]):
                confidence = person_detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                         continue

                    person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")

                    person_points = [startX, startY, endX, endY]

            # Desenhando a bola
            cv2.circle(frame, (ponto_circulo[0], ponto_circulo[1]), ponto_circulo[2], (255,0,255), thickness=2)

            # Detectando aproximação do jogador com a bola
            proximity = False
            if person_points:
                x1, y1, x2, y2 = [x for x in person_points]

                cX = int((x1 + x2) / 2.0)
                cY = int((y1 + y2) / 2.0)

                dx, dy = cX - ponto_circulo[0], y2 - ponto_circulo[1]

                distance = math.sqrt(dx * dx + dy * dy)
                distancia_minima = ponto_circulo[2] * 4

                if  distancia_minima <= distance <= distancia_minima + distancia_minima:
                    proximity = True

                if proximity:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            fps_end_time = datetime.datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)

            fps_text = "FPS: {:.2f}".format(fps)

            cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            cv2.imshow("Application", frame)
            frame = cv2.resize(frame, video_infos['dimensoes'])
            writer.write(frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            # import pdb; pdb.set_trace()

        cap.release()
        writer.release()

        cv2.destroyAllWindows()

def process_local_video(video_path):
    os.chdir(video_path)
    process_video()
    os.chdir(MAIN_PATH)

menus= {}
menus.update({
    'main': menu.Menu(
        "DISPARADOR DE SINAL NO MOMENTO DO CHUTE NA BATIDA DE PÊNALTI",
        'main', 
        f"""
{Y}[1]{B} Ler vídeo salvo localmente

{Y}[0]{B} Sair
        """,
        {0: [(exit, [])],
         1: [('change_menu', [1])],
         2: [(print, [f'{R}Método ainda não implementado !{B}'])],
        'file sys': [('change_menu', ['file_sys'])],
        }
    ),
})
file_sys_menu = menu.Menu(
        "Operações no diretório", 'file_sys',
        f"""
{Y}[1]{B} Ver diretórios disponíveis
{Y}[2]{B} Ver arquivos disponíveis
{Y}[3]{B} Ver diretório
{Y}[4]{B} Trocar de diretório

{Y}[0]{B} Voltar
        """,
        {0: [('change_prev_menu', [])],
         1: [(F_SYS.print_dir_itens, ['dirs'])],
         2: [(F_SYS.print_dir_itens, ['files'])],
         3: [(F_SYS.print_dir_itens, ['all'])],
         4: [(F_SYS.change_dir_action, [])],
        },
        parent_id=lambda self: self.call_id,
    )

menus.update({
    1: menu.Menu(
        nome="SELECIONAR VÍDEO LOCALMENTE", id=1, parent_id='main',
        msg=f"""
{Y}[1]{B} Escolher vídeos da RAW_VIDEOS
{Y}[2]{B} Escolher vídeos da RAW_VIDEOS2

{Y}[0]{B} Voltar
        """,
        methods={
            1: [(process_local_video, [SRC_PATH_RAW_VIDEOS])],
            2: [(process_local_video, [SRC_PATH_RAW_VIDEOS2])],
            0: [('change_parent_menu', [])]
        }
    ),
})

menus.update({
    'file_sys': file_sys_menu,
})


menu.limpar_tela()
if __name__ == '__main__':
    Display = menu.MenuDisplay(menus, 'main')
    running = True
    while running:
        menu_ativo = Display._get_menu_ativo()
        if not menu_ativo:
            print(f'{R}>>> TERMINANDO O PRORAMA ! \n>>> \n>>> O menu ativo não foi obtido{B}')
            running = False
            break
        print("\n==================================")
        print(f"{C}{os.getcwd()}{B}")
        Display.exec_menu()
