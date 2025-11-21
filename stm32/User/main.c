#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "robot.h"
#include "Key.h"

uint8_t i;

int main(void)
{
	robot_Init();    // 机器人初始化
	Key_Init();      // 按键初始化
	while (1)
	{
		if(Key_GetNum() == 1)
		{
		 makerobo_run(70,5000);//前进1S
		 makerobo_brake(500);//停止0.5S
   	}
		
	}
}
