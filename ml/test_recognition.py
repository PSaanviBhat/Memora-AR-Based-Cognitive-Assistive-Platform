def main():
    import cv2
    from detector import FaceDetector
    from recognizer import FaceRecognizer

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect_and_align(frame)
        if result:
            embedding, face = result
            name, score = recognizer.recognize(embedding)

            cv2.putText(
                frame,
                f"{name} ({score:.2f})",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Recognition Test", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
