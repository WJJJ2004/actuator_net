# -*- coding:utf-8 -*-
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QCursor, QFont
from PyQt5.QtWidgets import (
    QWidget, QTableView, QAbstractItemView, QToolTip, qApp, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QApplication, QDialog
)


class QTableViewPanel(QDialog):
    def __init__(self, users_data):
        super().__init__()
        self.column_name = list(users_data["column_name"])
        self.column_num = len(self.column_name)
        self.users_data = users_data["data"]
        self.row_num = users_data["row_num"]
        self.display_row_num = users_data["display_row_num"]
        print(users_data["data"][0])

        self.init_ui()
        self.main()

    def init_ui(self):
        """ UI 초기화 """
        self.resize(1500, 1000)  # 창 크기(너비, 높이) 설정
        self.setWindowTitle("원본 학습 데이터")  # 창 제목 설정
        self.setFixedSize(self.width(), self.height())

    def set_table_attribute(self):
        """ 창(테이블)의 각종 속성 설정 """
        self.set_table_column_row()
        self.set_table_header()
        self.set_table_init_data()
        self.set_table_v()
        # self.set_table_size()
        self.set_table_select()
        self.set_table_select_mode()

        self.set_table_header_visible()
        self.set_table_edit_trigger()
        self.show_table_grid()
        self.set_table_header_font_color()

    def set_table_column_row(self):
        """ 테이블의 행/열 개수 설정 """
        self.model = QStandardItemModel(self.display_row_num, self.column_num)

    def set_table_v(self):
        """ 테이블 뷰 생성 및 모델 연결 """
        self.table_view = QTableView()
        self.table_view.setModel(self.model)

    def set_table_header(self):
        """ 테이블 헤더(열 이름) 설정 """
        self.model.setHorizontalHeaderLabels(self.column_name)

    def set_table_init_data(self):
        """ 테이블 초기 데이터 입력 """
        for i in range(self.display_row_num):
            for j in range(self.column_num):
                user_info = QStandardItem(self.users_data[i][j])
                self.model.setItem(i, j, user_info)
                user_info.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    def set_table_size(self):
        """ 테이블 셀(열) 너비 설정 """
        self.table_view.setColumnWidth(0, 100)
        self.table_view.setColumnWidth(1, 130)
        self.table_view.setColumnWidth(2, 150)
        self.table_view.setColumnWidth(3, 150)
        self.table_view.setColumnWidth(4, 160)
        self.table_view.setColumnWidth(5, 165)

    def set_table_edit_trigger(self):
        """ 테이블 편집 가능 여부 설정(편집 불가) """
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)

    def set_table_select(self):
        """ 셀 선택 동작 설정 """
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectItems)

    def set_table_select_mode(self):
        """ 단일/다중 선택 설정 + 클릭/더블클릭 이벤트 연결 """
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_view.doubleClicked.connect(self.get_table_item)
        self.table_view.clicked.connect(self.get_cell_tip)

    def get_cell_tip(self):
        """ 셀 클릭 시 툴팁으로 전체 내용 표시 """
        contents = self.table_view.currentIndex().data()
        QToolTip.showText(QCursor.pos(), contents)

    def set_table_header_visible(self):
        """ 헤더 표시/숨김 설정 """
        self.table_view.verticalHeader().setVisible(True)
        self.table_view.horizontalHeader().setVisible(True)

    def get_table_item(self):
        """테이블에서 더블클릭한 셀의 내용을 복사"""
        # row = self.table_view.currentIndex().row()  # 행 번호
        column = self.table_view.currentIndex().column()  # 열 번호
        contents = self.table_view.currentIndex().data()  # 셀 데이터
        # QToolTip.showText(QCursor.pos(), contents)
        clipboard = qApp.clipboard()  # 클립보드
        clipboard.setText(contents)
        self.copy_tips1.setText("복사됨 (" + self.column_name[column] + "): ")
        self.copy_tips2.setText(contents)
        self.copy_tips2.setStyleSheet("color:red")

    def set_table_header_font_color(self):
        """ 헤더 글꼴/스타일 설정 """
        self.table_view.horizontalHeader().setFont(QFont("Verdana", 13, QFont.Bold))
        # self.table_view.horizontalHeader().setStyleSheet("")  # 스타일 추가 가능

    def show_table_grid(self):
        """ 테이블 격자선 표시 여부 """
        self.table_view.setShowGrid(True)

    def set_component(self):
        # self.btn_close = QPushButton("닫기")
        # self.btn_close.clicked.connect(self.close_window)  # 슬롯 연결

        self.label1 = QLabel("현재 총:")
        self.label_users_num = QLabel(str(self.row_num))
        self.label3 = QLabel("행 데이터! 셀을 더블클릭하면 해당 셀 내용이 복사됩니다!")
        self.copy_tips1 = QLabel("아직 복사한 내용이 없습니다!")
        self.copy_tips2 = QLabel()

        self.label_users_num.setStyleSheet("color:red")

        self.label1.setFixedWidth(42)
        self.label_users_num.setFixedWidth(100)
        # self.btn_close.setFixedSize(80, 32)
        self.copy_tips1.setFixedWidth(220)

    def set_panel_layout(self):
        """ 레이아웃 설정 """
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.table_view)

        h_layout1 = QHBoxLayout()
        h_layout1.addWidget(self.copy_tips1)
        h_layout1.addWidget(self.copy_tips2)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(self.label1)
        h_layout2.addWidget(self.label_users_num)
        h_layout2.addWidget(self.label3)

        # h_layout2.addWidget(self.btn_close)

        v_layout.addLayout(h_layout1)
        v_layout.addLayout(h_layout2)
        self.setLayout(v_layout)

    def close_window(self):
        self.close()
        qApp.quit()

    def main(self):
        self.set_table_attribute()
        self.set_component()
        self.set_panel_layout()


if __name__ == '__main__':
    user_data = {
        "data": [
            ["01", "철수", "행복로 1번지", "0.1", 'test1@qq.com', '활발하고 노래 부르는 것을 좋아함.'],
            ["02", "영희", "여기를 클릭하면 툴팁으로 전체 내용을 볼 수 있습니다", "13100000001", 'test2@qq.com', '낙천적이고, 남을 잘 돕는 편.'],
            ["03", "민수", "행복로 3번지", "13100000002", 'test3@qq.com', '여기도 표시 범위를 넘는 내용: 이 사람은 비교적 게을러서 소개를 안 적음.'],
            ["04", "지훈", "행복로 4번지", "13100000003", 'test4@qq.com', '말수가 적고 코딩을 좋아함.'],
            ["05", "수빈", "행복로 5번지", "13100000004", 'test5@qq.com', '긍정적이고 멍때리기를 좋아함.']
        ],
        "column_name": ["id", 'name', 'address', 'number', 'email', 'attr'],
        'row_num': 5,
        'display_row_num': 5
    }

    app = QApplication(sys.argv)
    tp = QTableViewPanel(user_data)
    tp.show()
    app.exit(app.exec_())
