#include "stm32f10x.h"
#include "Serial.h"

uint8_t  Vision_Mode = 0;      
// 0x01表示已经识别，0x00表示没有
int16_t  Vision_Error = 0;     
uint32_t Vision_Size = 0;      

static uint8_t RxState = 0;
static uint8_t RxCounter = 0;
static uint8_t RxBuffer[16]; 

void Serial_Init(void)
{
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
    RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

    GPIO_InitTypeDef GPIO_InitStructure;
    
    // TX  复用推挽输出
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_9;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF_PP;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_Init(GPIOA, &GPIO_InitStructure);
    
    // RX 浮空输入或上拉输入
    GPIO_InitStructure.GPIO_Pin = GPIO_Pin_10;
    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU; 
    GPIO_Init(GPIOA, &GPIO_InitStructure);

    USART_InitTypeDef USART_InitStructure;
    USART_InitStructure.USART_BaudRate = 115200; // 同步python baudrate
    USART_InitStructure.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
    USART_InitStructure.USART_Mode = USART_Mode_Tx | USART_Mode_Rx;
    USART_InitStructure.USART_Parity = USART_Parity_No;
    USART_InitStructure.USART_StopBits = USART_StopBits_1;
    USART_InitStructure.USART_WordLength = USART_WordLength_8b;
    USART_Init(USART1, &USART_InitStructure);

    USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);

    NVIC_InitTypeDef NVIC_InitStructure;
    NVIC_InitStructure.NVIC_IRQChannel = USART1_IRQn;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 1;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
    NVIC_Init(&NVIC_InitStructure);

    USART_Cmd(USART1, ENABLE);
}

void USART1_IRQHandler(void)
{
    uint8_t res;
	
    if (USART_GetITStatus(USART1, USART_IT_RXNE) != RESET)
    {
        res = USART_ReceiveData(USART1); 

        // 协议: packet = struct.pack('<BBhIB', 0xB3, 0x01或者0x00, x_offset, size, 0x5B)
        
        if (RxState == 0)
        {
            if (res == 0xB3)
            {
                RxState = 1;
                RxCounter = 0;
                RxBuffer[RxCounter] = res;
				RxCounter++;
            }
        }
        else if (RxState == 1)
        {
            RxBuffer[RxCounter] = res;
			RxCounter++;

            if (RxCounter >= 9)
            {
                if (RxBuffer[8] == 0x5B)
                {
                    Vision_Mode = RxBuffer[1];
                    
                    if (Vision_Mode == 0x01)
                    {
						// 小端是从后往前发送，从后往前读取
						
                        Vision_Error = (int16_t)(RxBuffer[2] | (RxBuffer[3] << 8));

                        Vision_Size = (uint32_t)(RxBuffer[4] | 
                                                (RxBuffer[5] << 8) | 
                                                (RxBuffer[6] << 16) | 
                                                (RxBuffer[7] << 24));
                    }
                    else
                    {
                        Vision_Error = 0;
                        Vision_Size = 0;
                    }
                }
                RxState = 0;
                RxCounter = 0;
            }
        }
    }
}