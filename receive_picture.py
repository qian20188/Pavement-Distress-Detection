import socket

def send_image(image_data):
    client = socket.socket()
    client.connect(('192.168.85.128', 21011))
    client.send(image_data)
    print('成功发送图像数据')
    client.close()

# 在服务器端接收图像数据并直接发送
server = socket.socket()
server.bind(('192.168.5.2', 21015))
server.listen(5)

while True:
    con, addr = server.accept()
    print('连接到: ', addr)

    # 接收图像数据
    image_data = b''
    while True:
        data = con.recv(1024)
        if not data:
            break
        image_data += data

    if image_data:
        send_image(image_data)  # 直接发送图像数据

    con.close()

server.close()