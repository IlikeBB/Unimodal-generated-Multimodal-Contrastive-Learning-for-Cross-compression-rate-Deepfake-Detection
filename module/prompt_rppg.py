# Ver.DFMv2
# 測試階段時會混合所有template
spoof_templates = [
    'This is an example of a forgery face',
    'This is an example of an attack face',
    'This is how a forgery face looks like',
    'a photo of a forgery face',
    'a printout shown to be a forgery face',
    "a synthetic face",
    "None",  # 無效文本
    "No information available",  # 無效文本
    "This text is irrelevant",  # 噪聲
    "Lorem ipsum dolor sit amet",  # 無意義文本
    "xyz123",  # 無意義字符串
    "随机的文本",  # 中文噪聲
    "Texto aleatorio",  # 西班牙文噪聲
    "Texte aléatoire",  # 法文噪聲
    "Random string for testing purposes",  # 英文噪聲
    "No description provided for this face"  # 與任務無關
]

real_templates = [
    'This is an example of a real face',
    'This is a bonafide face',
    'This is a real face',
    'This is how a real face looks like',
    'a photo of a real face',
    "a genuine human face",
    "None",  # 無效文本
    "No information available",  # 無效文本
    "This text is irrelevant",  # 噪聲
    "Lorem ipsum dolor sit amet",  # 無意義文本
    "xyz123",  # 無意義字符串
    "随机的文本",  # 中文噪聲
    "Texto aleatorio",  # 西班牙文噪聲
    "Texte aléatoire",  # 法文噪聲
    "Random string for testing purposes",  # 英文噪聲
    "No description provided for this face"  # 與任務無關
]