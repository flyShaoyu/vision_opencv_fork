import serial

# === 手动设置参数 ===
PORT = 'COM7'       # 修改为你的串口号(如COM3)
BAUD_RATE = 115200  # 波特率

# === 串口监听函数 ===
def listen():
    try:
        # 建立串口连接
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD_RATE,
            timeout=0.1  # 超时设置(秒)
        )
        print(f"监听开始: {PORT} @ {BAUD_RATE}bps")
        print("按Ctrl+C停止...\n")
        
        while True:
            # 尝试读取所有可用数据
            data = ser.read_all()
            
            if data:
                # 打印原始数据 (同时显示HEX和尝试解码)
                print(f"[原始] {data.hex().upper()}", end=' | ')
                try:
                    print(data.decode('utf-8', errors='replace').replace('\r', '\\r').replace('\n', '\\n'))
                except:
                    print("<解码失败>")
    
    except serial.SerialException as e:
        print(f"连接错误: 检查端口和驱动 | {str(e)}")
    except KeyboardInterrupt:
        print("\n监听停止")
    finally:
        ser.close() if 'ser' in locals() else None

# 启动监听
if __name__ == '__main__':
    print("=== CH340 最小化串口监听器 ===")
    print("检测到数据时将同时显示:")
    print("- HEX格式 (左侧)")
    print("- 尝试解码结果 (右侧)\n")
    listen()