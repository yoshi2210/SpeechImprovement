""" PySimpleGUIにより音声を認識し、テキスト音声合成する (PyAudio版) """
# cf. https://tam5917.hatenablog.com/entry/2021/09/06/234256#%E9%9F%B3%E5%A3%B0%E5%8F%8E%E9%8C%B2%E9%9F%B3%E5%A3%B0%E5%86%8D%E7%94%9F%E6%A9%9F%E8%83%BD


import threading
import numpy as np
import pyaudio
import PySimpleGUI as sg  # GUI構築のライブラリ
import wave
import sounddevice as sd  # 録音・再生系のライブラリ
import soundfile as sf  # 読み込み・書き出しのライブラリ
import speech_recognition as sr  # 音声認識のライブラリ

from scipy import interpolate
# import torch  # 深層学習のライブラリ
# from ttslearn.pretrained import create_tts_engine  # 音声合成ライブラリ


# Windowのサイズ (横, 縦) 単位ピクセル
WINDOW_SIZE = (600, 400)

# 出力先の音声ファイル名
# OUTPUT_FILE = "./tmp/record.wav"
OUTPUT_FILE = r"C:\Users\nakaj\Documents\record.wav"

FORMAT = pyaudio.paInt16
CHUNK = 512*2  # チャンクサイズ
SAMPLE_RATE = 44100 #16000  # サンプリング周波数
N_CHANNEL = 1  # チャンネル数 モノラルは1, ステレオは2
DURATION = 15.0  # 収録秒数
speed = 2 # 再生速度を通常の何倍にするか


# PyAudioのオブジェクト生成
PYAUDIO = pyaudio.PyAudio()

# PySimpleGUI 初期化
FONT = "Any 16"
sg.theme("SystemDefault1")



FRAME_STATUS = sg.Frame(
    layout=[
        [
            sg.Text(
                "ここにプログラム動作状況が表示されます",
                font=("Ricty", 12),
                text_color="#000000",
                background_color="#eee8d5",
                size=(50, 1),
                key="-STATUS_TEXT-",
            ),
        ],
    ],
    title="動作状況",
    font=("Ricty", 20),
    element_justification="left",
)

FRAME_RECOG = sg.Frame(
    layout=[
        [
            sg.Text(
                "ここに音声認識結果が表示されます",
                font=("Ricty", 12),
                text_color="#000000",
                background_color="#eee8d5",
                size=(50, 5),
                key="-RECOG_TEXT-",
            ),
        ],
    ],
    title="音声認識結果",
    font=("Ricty", 20),
    element_justification="left",
)

FRAME_ADVICE = sg.Frame(
    layout=[
        [
            sg.Text(
                "ここにアドバイスが表示されます",
                font=("Ricty", 12),
                text_color="#000000",
                background_color="#eee8d5",
                size=(50, 5),
                key="-ADVICE_TEXT-",
            ),
        ],
    ],
    title="音声分析結果",
    font=("Ricty", 20),
    element_justification="left",
)




# 各パーツのレイアウトを設定
# ウィンドウの下側に向かって、先頭から順に配置される
LAYOUT = [
    [FRAME_STATUS],
    [FRAME_RECOG],
    [FRAME_ADVICE],
    [
        # sg.ProgressBar(
        #     int(DURATION * SAMPLE_RATE / CHUNK),
        #     orientation="h",
        #     bar_color=("#dc322f", "#eee8d5"),
        #     size=(20, 22),
        #     key="-PROG-",
        # ),
        sg.Button("収録", key="-REC-", font=FONT),
        sg.FileBrowse(
            "開く",
            font=FONT,
            key="-FILES-",
            target="-FILES-",
            file_types=(("WAVEファイル", "*.wav")),
            enable_events=True,
        ),
        sg.Button("倍速再生", font=FONT, key="-PLAY-"),
        sg.Button("認識", key="-RECOG-", font=FONT),
        sg.Button("終了", font=FONT, key="-EXIT-"),
    ],
    
    # [
    #     FRAME_SPK,
    #     sg.Button("合成", key="-SYNTH-", target="-RECOG_TEXT-", font=("Ricty", 20)),
    # ],
]


WINDOW = sg.Window(
    "発表プレゼン矯正ツール", LAYOUT, finalize=True, size=WINDOW_SIZE
)


# 各関数からアクセスするグローバル変数
VARS = {
    "audio": None,  # 収録済み音声（もしくはロードした音声）
    "stream": None,  # 音声収録用ストリーム
}

# 各関数

def load_wav(file_name):
    """WAVファイルをロードする"""
    data, _ = sf.read(file_name)
    VARS["audio"] = data


def save_wav(file_name):
    """WAVファイルを保存する"""
    # 振幅の正規化
    audio = VARS["audio"] / np.abs(VARS["audio"]).max()
    audio = audio * (np.iinfo(np.int16).max / 2 - 1)
    audio = audio.astype(np.int16)

    sf.write(
        file=file_name,
        data=VARS["audio"],
        samplerate=SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )


def play_wav():
    """WAVを再生する"""
    if VARS["audio"] is None or len(VARS["audio"]) == 0:
        raise ValueError("Audio data must not be empty!")

    # 振幅の正規化
    audio = VARS["audio"] / np.abs(VARS["audio"]).max()
    audio = audio * (np.iinfo(np.int16).max / 2 - 1)
    audio = audio.astype(np.int16)

    # 音の高さが変わってしまうパターン
    # # 変換後のデータを格納する配列を用意
    # count = int((len(audio)-1)/speed)
    # print('count=',count)
    # dst = np.empty(count, dtype="int16")

    # # 補間関数を求める
    # f = interpolate.interp1d(range(len(audio)), audio, kind="cubic") 

    # # 再生速度変換
    # for i in range(0,count):
    #     dst[i] = f(i*speed)

    # print(SAMPLE_RATE*speed)
    # print(len(dst))

    # 音の高さ変えないように、時間窓で補間/抜粋するパターン
    # cf. https://toolbox.aaa-plaza.net/archives/3639
    windowsize=int(SAMPLE_RATE*0.05)   # 『窓』の幅
    count = int( (len(audio)-windowsize)/(windowsize*speed) )
    dst = np.zeros(int(len(audio)/speed), dtype="int16")

    for i in range(0,count):
        src_s = int(i*windowsize*speed)
        dst_s = i*windowsize

        for i in range(windowsize):
            dst[dst_s+i] = audio[src_s+i]


    # 再生
    sd.play(dst, SAMPLE_RATE)

    # 再生は非同期に行われるので、明示的にsleepさせる
    sd.sleep(int(1000 * len(VARS["audio"]) / SAMPLE_RATE))


def play_stop():
    """WAV再生を停止する"""
    if VARS["audio"] is None or len(VARS["audio"]) == 0:
        raise ValueError("Audio data must not be empty!")

    sd.stop()


def progress_bar():
    """プログレスバーを表示する関数"""

    # 録音している間、プログレスバーを並行して表示
    for i in range(int(DURATION * SAMPLE_RATE / CHUNK)):
        # プログレスバーを更新
        WINDOW["-PROG-"].update(i + 1)


def record():
    """プログレスバーの進行と非同期に録音する関数
    Threading による非同期実行
    """
    WINDOW["-STATUS_TEXT-"].Update('収録開始')
    # 入力ストリームを開く
    VARS["stream"] = PYAUDIO.open(
        format=FORMAT, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK
    )
    print('a')
    WINDOW["-STATUS_TEXT-"].Update('収録データの格納開始')
    # ストリームから音声を取得
    frames = []  # 連結用リスト
    for _ in range(int(DURATION * SAMPLE_RATE / CHUNK)):

        # ストリームからCHUNKサイズだけ音声データを取得
        frame = VARS["stream"].read(CHUNK, exception_on_overflow=False)

        # 配列を一時的にリストに保存
        frames.append(frame)

    # 連結
    frames = b"".join(frames)
    print('b')

    # データをNumpy配列に変換
    recording = np.frombuffer(frames, dtype="int16")

    # 収録結果を保存
    VARS["audio"] = recording
    print('c')
    WINDOW["-STATUS_TEXT-"].Update('収録データ格納終了')


    # ストリームの終了
    VARS["stream"].stop_stream()
    VARS["stream"].close()


def listen():
    """リッスンする関数"""

    # 音声録音の非同期実行
    record_thread = threading.Thread(target=record, daemon=True)
    record_thread.start()  # 終了すると自動でterminateする

    # # 録音している間、プログレスバーを並行して表示
    # for i in range(int(DURATION * SAMPLE_RATE / CHUNK)):
    #     # プログレスバーを更新
    #     WINDOW["-PROG-"].update(i + 1)

    # # プログレスバーのクリア
    # WINDOW["-PROG-"].update(0)


def recog():
    """音声認識する関数"""
    print('aaa')
    # 振幅の正規化
    audio = VARS["audio"] / np.abs(VARS["audio"]).max()
    audio = audio * (np.iinfo(np.int16).max / 2 - 1)
    audio = audio.astype(np.int16)

    print(len(audio))
    print(audio[100])
    print('aaa')
    print(OUTPUT_FILE)
    
    # 一旦ファイルに書き込む
    sf.write(
        file=OUTPUT_FILE,
        data=VARS["audio"],
        samplerate=SAMPLE_RATE,
        format="WAV",
        subtype="PCM_16",
    )

    WINDOW["-STATUS_TEXT-"].Update('音声認識開始')
    r = sr.Recognizer()
    print('aaa')
    with sr.AudioFile(OUTPUT_FILE) as source:
        audio = r.listen(source)  # 音声取得
        print('acquire')
        text = r.recognize_google(audio, language="ja-JP")
        print('textized')
        WINDOW["-RECOG_TEXT-"].Update(text)
    WINDOW["-STATUS_TEXT-"].Update('音声認識終了')
    WINDOW["-ADVICE_TEXT-"].Update('70点：若干話すスピードが早いです。もっとゆっくり重低音を出して、威厳を出していくとよいかもしれませんね。語末に吊り上がる癖もやめた方がよいと思われます。')

    


def event_play_record(event):
    """再生・録音系のイベント処理"""

    # 音声を再生
    if event == "-PLAY-":
        play_wav()

    # 再生を停止
    elif event == "-STOP-":
        play_stop()

    # 録音
    elif event == "-REC-":
        listen()


def event_recog(event, values):
    """音声認識系のイベント処理"""
    if event == "-RECOG-":
        print('here')
        recog()


def event_synth(event, values):
    """音声合成系のイベント処理"""
    if event == "-SYNTH-":
        text = WINDOW["-RECOG_TEXT-"].DisplayText

        # テキスト音声合成
        wav, sr = PWG_ENGINE.tts(text, spk_id=SPK_ID)

        # 音割れ防止
        wav = (wav / np.abs(wav).max()) * (np.iinfo(np.int16).max / 2 - 1)

        # 再生
        sd.play(wav.astype(np.int16), sr)
        sd.sleep(int(1000 * len(wav) / sr))


def change_spk(spk) -> None:
    """話者変更."""
    global SPK_ID
    SPK_ID = PWG_ENGINE.spk2id[spk]
    WINDOW["SPK_NAME"].Update("現在の話者：{}".format(SPK2ID[spk]))


def finalize():
    """終了処理"""

    # PyAudioを閉じる
    PYAUDIO.terminate()

    # Windowを閉じる
    WINDOW.close()



def mainloop():
    """メインのループ"""

    while True:  # 無限ループにすることでGUIは起動しつづける
        event, values = WINDOW.read()  # イベントと「値」を取得
        print(event, values)  # debug

        # windowを閉じるか 終了ボタンを押したら終了
        if event in (sg.WIN_CLOSED, "-EXIT-"):
            finalize()
            break

        # 再生・録音系イベント
        if event in ("-PLAY-", "-STOP-", "-REC-"):
            event_play_record(event)

        # ファイルオープンイベント
        elif event in ("-FILES-"):
            if values["-FILES-"] != "":
                load_wav(values["-FILES-"])

        # 音声認識
        elif event in ("-RECOG-"):
            event_recog(event, values)

        # テキスト音声合成
        elif event in ("-SYNTH-"):
            event_synth(event, values)



if __name__ == "__main__":
    # GUI起動
    print('GUI起動')
    mainloop()
