R = '\033[31m' # vermelho
G = '\033[01;32m' # verde
Y = '\033[01;33m' # amarelo
C = '\033[36m' # ciano
W = '\033[0m'  # branco

import os
import os.path as path


IGNORES_EXT = []

def ignore_item(name):
    """ Função para verificar pelo nome do arquivo se ele deve
    ser ignorado. As verificações são se ele é um arquivo oculto
    ou se a extensão do arquivo termina com alguma extensão a ser
    ignorada.

    Args:
        name (string): Nome do Arquivo.

    Returns:
        bool: Verdadeiro se o arquivo deve ser ignorado
    """
    ignore = name.startswith('.') or any([name.endswith(ig) for ig in IGNORES_EXT])
    return ignore

def get_directory_itens(pth, recursive=True, files_only=True):
    """ Método para obter os itens do diretório atual aplicando as configurações de pesquisa.
    As configurações utilizadas são:
      - 'recursive_find': para determinar se deve ser observado também os arquivos nas pastas encontradas.
         com essa config habilitada, a função é chamada recursivamente para cada pasta encontrada e assim por diante.
      - 'include_dirs': para saber se os diretórios devem ser mostrados na busca.

    Args:
        pth (str): caminho do diretório a ser scaneado
        recursive (bool): Indica se devem ser buscados arquivos em subpastas do diretório
        files_only (str): caminho do diretório a ser scaneado

    Returns:
        list: itens scaneados no diretório 'pth'
    """
    itens = []
    for scan in os.scandir(pth):
        ignore = ignore_item(scan.name)
        if not ignore:
            if scan.is_dir() and recursive:
                itens += get_directory_itens(path.join(pth, scan.name))
            if scan.is_file() or scan.is_dir() and not files_only:
                itens.append(scan)
    return itens

def print_dirs(itens=[], recursive=False):
    """Mostrar diretórios do diretório

    Args:
        itens (list, optional): Itens a serem olhados.
        recursive (bool, optional): Se devem ser olhados os diretórios recursivamente.
    """
    # Filtra apenas os itens que são diretórios
    dirs = filter(lambda item: item.is_dir(), itens or get_directory_itens('.', recursive, False))
    print(f"{C} ---  DIRETÓRIOS  ---{W}")
    for d in dirs:
        print(f"{G}-> {Y}{d.name}{W}")
    print()

def print_files(itens=[]):
    """Mostrar arquivos do diretório
    """
    if not itens:
        itens = os.scandir()
    # Filtra apenas os itens que são arquivos e os ordena pelo tamanho
    files = sorted(filter(lambda item: item.is_file(), itens), key=lambda f: path.getsize(f.path))
    print(f"{C} ---  ARQUIVOS  ---{W}")
    for f in files:
        if path.exists(f.path):
            print(f"{Y}-> {f.name}   |   {G}{path.getsize(f.path) / 1024} kB{W}")
    print(" -------------------------------------- ")

def print_dir_itens(opt='all', pth='.', recursive=False):
    """ Imprime na tela os diretórios e/ou arquivos de um diretório especifico.

    Args:
        opt (str): 'all' para mostrar arquivos e diretórios 'dirs' ou 'files' 
            para somente arquivos ou diretórios.
        
        pth (str): Caminho do diretório que deve ser explorado.
    """
    # Obtém os itens do diretório especificado
    dir_itens = get_directory_itens(pth, recursive, files_only= opt=='files')

    # Imprime na tela os diretórios e/ou arquivos no caminho específico
    if opt in ['all', 'dirs']:
        print_dirs(dir_itens)
    if opt in ['all', 'files']:
        print_files(dir_itens)

def validate_dir_change(pth):
    exists = path.exists(pth)
    if not exists:
        print(f'{R}Não foi possível encontrar o caminho "{W}{pth}{R}"{W}')
    return exists

def change_dir_action():
    print(f"{G}Escolha um diretório a seguir para acessar:{W}")
    print_dir_itens('dirs')
    try:
        ask = True
        while ask:
            dir_name = input(f"Nome: {G}")
            valid = validate_dir_change(dir_name)
            if valid:
                os.chdir(dir_name)
                print(f'{Y}O diretório foi alterado para: {C}{os.getcwd()}{W}\n')
                ask = False
    # Pressione CTRL+C para voltar ao menu 
    except KeyboardInterrupt:
        print('Voltando ao menu ...')


def pedir_nome_video():
    print_dir_itens('files')
    print(f'{G}Escolha um dos arquivos acima para realizar a leitura do vídeo{W}')
    valid = False
    nome = ''
    while not valid:
        try:
            nome = input(f'{G}>> ')
            print(f'{W}', end='')
            valid = path.exists(nome)
            if not valid:
                print(f'{Y}Não foi possível encontrar o vídeo com o nome "{C}{nome}{Y}"{W}')
                print(f'{G}Digite novamente ou CTRL+C para voltar ao menu anterior{W}', end='\n\n')
        except KeyboardInterrupt:
            return False
    return path.abspath(path.join('.', nome))
