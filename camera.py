import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera index {i} available.")
        cap.release()
cap = cv2.VideoCapture(1)  # ganti dengan index iPhone camera kamu
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("iPhone Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
