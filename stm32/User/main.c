#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "robot.h"
#include "Key.h"
#include "LED.h"
#include "Serial.h" 

float Kp = 0.25f;           // 稍微加大一点，防止低速带不动
float Kd = 0.0f;           
float Prev_Error = 0;         // 上一次误差

int Base_Speed = 100;       
uint8_t Run_Mode = 1;       // 0待机, 1视觉循迹

int main(void)
{
	robot_Init();    
	Key_Init();     
	LED_Init();      
	Serial_Init();  
	
	    // 轮子微转 (左轮转，右轮转)
    makerobo_SetPWM(80, 80); 
    Delay_ms(200);           // 转0.2秒
    makerobo_SetPWM(0, 0);   // 停
    Delay_ms(200);           // 停0.2秒
    makerobo_SetPWM(80, 80);  // 反转
    Delay_ms(200);
    makerobo_SetPWM(0, 0);   // 彻底停下

	LED1_ON();       
	Delay_ms(1000);
	LED1_OFF();

	while (1)
	{
		if(Key_GetNum() == 1)
		{
			Run_Mode = !Run_Mode; 
			if (Run_Mode==1) 
			{
				LED2_ON();
			} 
			else 
			{
				LED2_OFF();
				makerobo_SetPWM(0, 0); 
			}
		}

		if (Run_Mode == 1)
		{
			if (Vision_Mode == 0)
			{
				makerobo_SetPWM(0, 0);
				Prev_Error = 0; 
			}
			else 
			{
				if (Vision_Size > 85000)
				{
					makerobo_SetPWM(0, 0);
					Prev_Error = 0;
				}
				else
				{
                                        float pid_output = (Kp * -Vision_Error) + (Kd * (-Vision_Error - Prev_Error));
                                        Prev_Error = -Vision_Error;

					int turn_val = (int)pid_output;

					int left_motor  = Base_Speed + turn_val;
					int right_motor = Base_Speed - turn_val;

					makerobo_SetPWM(left_motor, right_motor);
				}
			}
		}
		
		Delay_ms(10);
	}
}