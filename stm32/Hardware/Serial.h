#ifndef __SERIAL_H
#define __SERIAL_H

#include <stdint.h>

extern uint8_t  Vision_Mode;
extern int16_t  Vision_Error;
extern uint32_t Vision_Size;

void Serial_Init(void);

#endif