#include "stm32f10x.h"                  // Device header
#include "PWM.h"
#include "Delay.h"

void robot_Init(void)
{
	PWM_Init(); 
}

    /*
     * The right motor wiring is reversed, so a positive logical value should
     * drive the wheel forward with the opposite PWM polarity.
     */
    right_pwm = -right_pwm;

//四路PWM控制速度调节，1speed前进，2speed后退（永远是正数，同时另一个必须设为0）
void robot_speed(uint8_t left1_speed,uint8_t left2_speed,uint8_t right1_speed,uint8_t right2_speed)
{	
    TIM_SetCompare1(TIM4,left1_speed);
    TIM_SetCompare2(TIM4,left2_speed);
    TIM_SetCompare3(TIM4,right1_speed);
    TIM_SetCompare4(TIM4,right2_speed);
}

/**
  * @brief  支持正负速度的运动控制
  * @param  left_pwm:  左轮速度 -100到100
  * @param  right_pwm: 右轮速度 -100到100
  */
void makerobo_SetPWM(int16_t left_pwm, int16_t right_pwm)
{
    uint8_t L1=0, L2=0, R1=0, R2=0;

    if (left_pwm >= 0) 
    {
        if(left_pwm > 100) left_pwm = 100;
        L1 = left_pwm;
        L2 = 0;
    }
    else 
    {
        if(left_pwm < -100) left_pwm = -100;
        L1 = 0;
        L2 = -left_pwm; 
    }

    if (right_pwm >= 0) 
    {
        if(right_pwm > 100) right_pwm = 100;
        R1 = right_pwm;
        R2 = 0;
    }
    else 
    {
        if(right_pwm < -100) right_pwm = -100;
        R1 = 0;
        R2 = -right_pwm; 
    }

    robot_speed(L1, L2, R1, R2);
}




// 基本的运动函数
// 机器人前进
void makerobo_run(int8_t speed,uint16_t time)  //前进函数
{
        if(speed > 100)
		{
			speed = 100;
		}
		if(speed < 0)
		{
			speed = 0;
		}
	    robot_speed(speed,0,speed,0);
		Delay_ms(time);                 // 时间为毫秒
		robot_speed(0,0,0,0);           // 机器人停止
}

void makerobo_brake(uint16_t time) //刹车函数
{
		robot_speed(0,0,0,0);     // 电机停止 
		Delay_ms(time);          // 时间为毫秒    
}

void makerobo_Left(int8_t speed,uint16_t time) //左转函数
{
	    if(speed > 100)
			{
				speed = 100;
			}
			if(speed < 0)
			{
				speed = 0;
			}
		robot_speed(0,0,speed,0);
		Delay_ms(time);                 //时间为毫秒  
	  robot_speed(0,0,0,0);           // 机器人停止

}

void makerobo_Spin_Left(int8_t speed,uint16_t time) //左旋转函数
{
		  if(speed > 100)
			{
				speed = 100;
			}
			if(speed < 0)
			{
				speed = 0;
			}  
		robot_speed(0,speed,speed,0);
		Delay_ms(time);                    //时间为毫秒 
    robot_speed(0,0,0,0);           // 机器人停止			
}

void makerobo_Right(int8_t speed,uint16_t time) //右转函数
{
	    if(speed > 100)
			{
				speed = 100;
			}
			if(speed < 0)
			{
				speed = 0;
			}
		robot_speed(speed,0,0,0);
		Delay_ms(time);                 //时间为毫秒  
	  robot_speed(0,0,0,0);           // 机器人停止

}

void makerobo_Spin_Right(int8_t speed,uint16_t time) //右旋转函数
{
		  if(speed > 100)
			{
				speed = 100;
			}
			if(speed < 0)
			{
				speed = 0;
			}  
		robot_speed(speed,0,0,speed);
		Delay_ms(time);                    //时间为毫秒 
    robot_speed(0,0,0,0);           // 机器人停止			
}

void makerobo_back(int8_t speed,uint16_t time)  //后退函数
{
      if(speed > 100)
			{
				speed = 100;
			}
			if(speed < 0)
			{
				speed = 0;
			}
	    robot_speed(0,speed,0,speed);
			Delay_ms(time);                 // 时间为毫秒
			robot_speed(0,0,0,0);           // 机器人停止
 
}