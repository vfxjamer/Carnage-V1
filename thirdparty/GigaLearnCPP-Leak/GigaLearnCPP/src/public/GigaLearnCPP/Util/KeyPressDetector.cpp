#include "KeyPressDetector.h"

#ifdef _MSC_VER
#include <conio.h>

char GGL::KeyPressDetector::GetPressedChar() {
	return _getch();
}

#else
// Non-MSVC (Linux/Colab): no interactive key reads. Training is controlled via
// signals (or the notebook's STOP cell), not keyboard input. Calling tcsetattr
// on a non-TTY stdin (as done previously) spams "Inappropriate ioctl" errors.
char GGL::KeyPressDetector::GetPressedChar() {
	return 0;
}

#endif