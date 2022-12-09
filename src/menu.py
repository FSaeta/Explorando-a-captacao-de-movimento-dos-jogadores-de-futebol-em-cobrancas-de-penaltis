from menu_utils import R, G, B, C, Y
import menu_utils as MUtils

from os import system, name

def limpar_tela():
    system('cls' if name == 'nt' else 'clear')

def define_contexto(menus, id, id_pai):
    menu = menus[id]
    menu_id = str(id_pai)
    if str(id_pai) != '0':
        new_contexto = menus[id_pai].contexto['msg'] + " > " + menu.nome
        menu.contexto.update({
            'msg': f"{G}{new_contexto}{B}",
            'id_pai': id_pai
        })

class Menu:
    def __init__(self, nome, id, msg, methods, parent_id=False):
        self.nome = nome
        self.id = id
        self.msg = msg
        self.methods = methods
        self.funcoes_params = self.get_funcoes_params()
        self.contexto = {'msg': f"{G}-> {self.nome}{B}", 'id_pai': 0}
        self.parent_id = parent_id
        self.parent_call = False

    def get_funcoes_params(self):
        funcoes_parametros = {}
        for opcao in self.methods.keys():
            funcs_params = []
            for funcao_params in self.methods[opcao]:
                funcs_params.append(funcao_params)

            funcoes_parametros.update({opcao:funcs_params})
        return funcoes_parametros

    def add_funcs(self, opcao, funcao, param):
        """Adicionar funções em tempo de execução ou em partes do 
        código após a declaração do menu

        - Formato de exemplo: menu.add_funcs(1, limpar_tela, [])"""

        funcao_parametro = (funcao,param)
        self.methods.update({opcao, funcao_parametro})

    def mostrar_msg(self, add_msg=False):
        if self.contexto['msg']:
            print(self.contexto['msg'])
        if add_msg:
            print('\n' + add_msg)
        print(self.msg)

    def pede_opcao(self):
        opcoes = list(self.methods.keys())
        while True:
            op = input(f"Opção: {G}")
            if op.isnumeric():
                op = int(op)
            print(f"{B}")
            if op in opcoes:
                return op
            print(f"{R}Opção Inválida !{B}")


class MenuDisplay:
    
    def __init__(self, menus={}, force_id=False):
        self.menus = menus

        # Definindo um ID principal
        if force_id and force_id in menus.keys():
            menu_id = force_id
        else:
            menu_id = 0 if not menus else menus[[x for x in menus.keys()][0]]

        self.active_id = menu_id
        self.prev_id = menu_id


    def _add_menu(self, menu, id):
        """ Inclui um novo menu na tela de menus

        Args:
            menu (Menu): menu que sera incluído aos menus
            id (any): id para acessar o menu via dicionário
        """
        self.menus.update({id: menu})

    def _get_menu(self, menu_id):
        """ Obtém o menu de id 'menu_id' caso exista, caso contrário,
        avisa que o menu não foi encontrado e retorna False
        """
        menu =  self.menus.get(menu_id)
        if menu:
            return menu
        else:
            print(f"{R}Nenhum menu com o id {C}'{menu_id}'{R} foi encontrado{B}")
            return False

    def _get_menu_ativo(self):
        """Obtém o menu ativo
        """
        return self._get_menu(self.active_id)

    def change_menu(self, menu_id):
        """ Realiza a troca do menu ativo e salva o menu ativo como o 
        menu anterior

        Args:
            menu_id (any): id do menu a ser acessado
        """
        if self._get_menu(menu_id):
            self.prev_id = self.active_id
            self.active_id = menu_id

    def change_prev_menu(self):
        """ Altera pro menu anterior
        """
        atual = self.active_id
        self.active_id = self.prev_id
        self.prev_id = atual
    
    def change_parent_menu(self, *params):
        """ Altera para o menu Pai se houver
        """
        menu = self._get_menu_ativo()
        if menu and menu.parent_id:
            if '__call__' in dir(menu.parent_id):
                id_menu = menu.parent_id(*params)
            else:
                id_menu = menu.parent_id
            self.change_menu(id_menu)
        else:
            print(f"{R}Nenhum menu a cima do menu {C}'{menu.nome}'{R} foi encontrado{B}")

    def exec_menu_funcs(self, opcao):
        """Executa todas as funções que o menu possui para a opção escolhida"""
        ret = {}
        menu = self._get_menu_ativo()

        for funcao, params in menu.funcoes_params[opcao]:
            if type(funcao) == str:
                # Caso a função passada seja uma string, é buscado um método no objeto de menu 
                if hasattr(self, funcao):
                    func = getattr(self, funcao)
                    ret_f = func(*params) if params else func()
                else:
                    print(f"{R} FUNÇÃO '{funcao}' NÃO EXISTE !{B}")
                    ret_f = 'Erro'
                func_name = funcao
            else:
                if funcao != exit:
                    func_name = funcao.__name__
                ret_f = funcao(*params) if params else funcao()
            ret.update({func_name: ret_f})
        return ret

    def exec_menu(self):
        menu = self._get_menu_ativo()
        menu.mostrar_msg()
        opcao = menu.pede_opcao()
        ret = self.exec_menu_funcs(opcao)

