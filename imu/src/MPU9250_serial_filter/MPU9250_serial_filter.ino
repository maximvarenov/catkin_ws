#include "quaternionFilters.h"
#include "MPU9250.h"
#include "helper_3dmath.h"
#ifdef LCD
#include <Adafruit_GFX.h>
#include <Adafruit_PCD8544.h>

Adafruit_PCD8544 display = Adafruit_PCD8544(9, 8, 7, 5, 6);
#endif // LCD

#define AHRS true         // Set to false for basic data read
#define SerialDebug true  // Set to true to get Serial output for debugging

//#########################################################
// Noise filter variables
VectorInt16 noiseParaAccel(10,10,10);          //parameter accelerometer
VectorInt16 noiseParaAccelSen(0.0898, 0.0898, 0.0898);       //parameter accelerometer
float noiseParaAccelLow = 0.05;                   //parameter accelerometer low pass filter
VectorInt16 noiseParaGyro(0.3115, 0.3115, 0.3115);              //parameter gyroscope
VectorInt16 noiseParaGyroSen(0.267, 0.267, 0.267);           //parameter gyroscope
float noiseParaGyroLow = 0.05;                       //parameter gyroscope low pass filter
VectorInt16 noiseParaMag(6, 6, 6);                  //parameter magnetometer
VectorInt16 noiseParaMagSen(6, 6, 6);               //parameter magnetometer
float noiseParaMagLow = 0.05;                      //parameter magnetometer low pass filter
VectorFloat nullVector = VectorFloat(0, 0, 0);      //null vector for reference

// Low pass filter variables
VectorFloat new_afilt = VectorFloat(0,0,0);
VectorFloat old_afilt;
VectorFloat new_aLP = VectorFloat(0, 0, 0);
VectorFloat old_aLP;
VectorFloat new_vfilt = VectorFloat(0,0,0);
VectorFloat old_vfilt;
VectorFloat new_vLP = VectorFloat(0, 0, 0);
VectorFloat old_vLP;
VectorFloat new_sfilt = VectorFloat(0,0,0);
VectorFloat old_sfilt;
VectorFloat new_sLP = VectorFloat(0, 0, 0);
VectorFloat old_sLP;

// Orientation/motion variables
VectorInt16 gyroReadout;                            // [x, y, z]     gyro sensor measurements
VectorFloat gyroFilteredF;                          // [x, y, z]     gyro sensor measurements (Float)
VectorInt16 aaReadout;                              // [x, y, z]     accel sensor measurements
VectorFloat aaReadoutF;                             // [x, y, z]     accel sensor measurements (Float)
VectorInt16 aaFiltered;                             // [x, y, z]     accel sensor measurements noise filtered 
VectorFloat aaFilteredF;
VectorFloat gravity;                                // [x, y, z]     gravity vector
VectorInt16 magReadout;                             // [x, y, z]     magnetometer Readout
VectorFloat magReadoutF;                            // [x, y, z]     magnetometer Readout in floats
VectorFloat magFilteredF;                           // [x, y, z]     magnetometer filtered in floats
VectorFloat magNormalizedF;                         // [x, y, z]     magnetometer normalized in floats

float gravityx;  //gravity 
float gravityy;
float gravityz;

// Translation variables
float dT;                                           //secondes       duration of calculation interval
VectorFloat old_a;
VectorFloat new_a;
VectorFloat old_v;
VectorFloat new_v;
VectorFloat old_s;
VectorFloat new_s;

VectorFloat noiseFilter(VectorFloat *compareValue, VectorInt16 *value, VectorFloat *previousCorrect, float noiseParameterLowPass, VectorInt16 noiseParameterSen, VectorInt16 *noiseParameter  );
float trapezoid(float new_data, float old_data, float old_out, float dT);
VectorFloat lowpassfilter(VectorFloat *prevX, VectorFloat *newU, VectorFloat *prevU, float alpha);
int count;
//##########################################################################

// Pin definitions
int intPin = 12;  // These can be changed, 2 and 3 are the Arduinos ext int pins
int myLed  = 13;  // Set up pin 13 led for toggling

#define I2Cclock 400000 //400khz
#define I2Cport Wire
#define MPU9250_ADDRESS MPU9250_ADDRESS_AD0   // Use either this line or the next to select which I2C address your device is using

MPU9250 myIMU(MPU9250_ADDRESS, I2Cport, I2Cclock);

void setup()
{
  Wire.begin();
  TWBR = 12;  // 400 kbit/sec I2C speed
  Serial.begin(115200);
  while(!Serial){};
  // Set up the interrupt pin, its set as active high, push-pull
  pinMode(intPin, INPUT);
  digitalWrite(intPin, LOW);
  pinMode(myLed, OUTPUT);
  digitalWrite(myLed, HIGH);
  byte c = myIMU.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);
  if (c == 0x71) // WHO_AM_I should always be 0x71
  {
    //Serial.println(F("MPU9250 is online..."));
    // Start by performing self test and reporting values
    myIMU.MPU9250SelfTest(myIMU.selfTest);
    myIMU.calibrateMPU9250(myIMU.gyroBias, myIMU.accelBias);
    myIMU.initMPU9250();
    byte d = myIMU.readByte(AK8963_ADDRESS, WHO_AM_I_AK8963);
   
    if (d != 0x48)
    {
      // Communication failed, stop here
      //Serial.println(F("Communication failed, abort!"));
      Serial.flush();
      abort();
    }
    // Get magnetometer calibration from AK8963 ROM
    myIMU.initAK8963(myIMU.factoryMagCalibration);
    myIMU.getAres();
    myIMU.getGres();
    myIMU.getMres();
    myIMU.magCalMPU9250(myIMU.magBias, myIMU.magScale);
  } 
  else
  {
    // Communication failed, stop here
    //Serial.println(F("Communication failed, abort!"));
    Serial.flush();
    abort();
  }
}

// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================

void loop()
{
  // If intPin goes high, all data registers have new data
  // On interrupt, check if data ready interrupt
  if (myIMU.readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01)
  {
    myIMU.readAccelData(myIMU.accelCount); 
    myIMU.ax = 9.81*(float)myIMU.accelCount[0] * myIMU.aRes ;//- myIMU.accelBias[0];
    myIMU.ay = 9.81*(float)myIMU.accelCount[1] * myIMU.aRes ;//- myIMU.accelBias[1];
    myIMU.az = 9.81*(float)myIMU.accelCount[2] * myIMU.aRes ;//- myIMU.accelBias[2];
    // in m/s2
  
    myIMU.readGyroData(myIMU.gyroCount);  
    myIMU.gx = DEG_TO_RAD*((float)myIMU.gyroCount[0] * myIMU.gRes - myIMU.gyroBias[0]);
    myIMU.gy = DEG_TO_RAD*((float)myIMU.gyroCount[1] * myIMU.gRes - myIMU.gyroBias[1]);
    myIMU.gz = DEG_TO_RAD*((float)myIMU.gyroCount[2] * myIMU.gRes - myIMU.gyroBias[2]);
    // in rad/s

    // Calculate the magnetometer values in milliGauss
    myIMU.readMagData(myIMU.magCount); 
    myIMU.mx = ((float)myIMU.magCount[0] * myIMU.mRes
               * myIMU.factoryMagCalibration[0] - myIMU.magBias[0]);
    myIMU.my = ((float)myIMU.magCount[1] * myIMU.mRes
               * myIMU.factoryMagCalibration[1] - myIMU.magBias[1]);
    myIMU.mz = ((float)myIMU.magCount[2] * myIMU.mRes
               * myIMU.factoryMagCalibration[2] - myIMU.magBias[2]);

  }      
        
    aaReadout.x = myIMU.ax ;
    aaReadout.y = myIMU.ay ;
    aaReadout.z = myIMU.az ;
    gyroReadout.x = myIMU.gx;
    gyroReadout.y = myIMU.gy;
    gyroReadout.z = myIMU.gz;
    
    aaFilteredF   = noiseFilter(&aaFilteredF,   &aaReadout,   &aaFilteredF,  noiseParaAccelLow, &noiseParaAccelSen, &noiseParaAccel);
    gyroFilteredF = noiseFilter(&gyroFilteredF, &gyroReadout, &nullVector,   noiseParaGyroLow,  &noiseParaGyroSen,  &noiseParaGyro);
   
    // Must be called before updating quaternions!
    myIMU.updateTime();
    MadgwickQuaternionUpdate(aaFilteredF.x, aaFilteredF.y, aaFilteredF.z, myIMU.gx * DEG_TO_RAD,
                         myIMU.gy * DEG_TO_RAD, myIMU.gz * DEG_TO_RAD, myIMU.my,
                         myIMU.mx, myIMU.mz, myIMU.deltat);         
    myIMU.delt_t = millis() - myIMU.count;
    gravityx = 2 * (*(getQ() + 1) * *(getQ() + 3) - *getQ() * *(getQ() + 2));
    gravityy = 2 * (*getQ() * *(getQ() + 1) + *(getQ() + 2) * *(getQ() + 3));
    gravityz = *getQ() * *getQ() - *(getQ() + 1)* *(getQ() + 1) - *(getQ() + 2) * *(getQ() + 2) + *(getQ() + 3)* *(getQ() + 3);


/*/##################################################################
    gravityx = 2 * (*(getQ() + 1) * *(getQ() + 3) - *getQ() * *(getQ() + 2));
    gravityy = 2 * (*getQ() * *(getQ() + 1) + *(getQ() + 2) * *(getQ() + 3));
    gravityz = *getQ() * *getQ() - *(getQ() + 1)* *(getQ() + 1) - *(getQ() + 2) * *(getQ() + 2) + *(getQ() + 3)* *(getQ() + 3);

    aaReadout.x = myIMU.ax - gravityx *9.81;
    aaReadout.y = myIMU.ay - gravityy *9.81;
    aaReadout.z = myIMU.az - gravityz *9.81;
     
    aaFilteredF   = noiseFilter(&aaFilteredF,   &aaReadout,   &aaFilteredF,  noiseParaAccelLow, &noiseParaAccelSen, &noiseParaAccel);
   
    myIMU.ax = aaFilteredF.x;
    myIMU.ay = aaFilteredF.y;
    myIMU.az = aaFilteredF.z;
    
//##################################################################    
*/
  
  if (!AHRS)
  {
    myIMU.delt_t = millis() - myIMU.count;
    if (myIMU.delt_t > 14)
    {
      if(SerialDebug)
      {
        myIMU.tempCount = myIMU.readTempData();  // Read the adc values
        // Temperature in degrees Centigrade
        myIMU.temperature = ((float) myIMU.tempCount) / 333.87 + 21.0;
        //Serial.print("Temperature is ");  Serial.print(myIMU.temperature, 1);
        //Serial.println(" degrees C");
      }

      myIMU.count = millis();
      digitalWrite(myLed, !digitalRead(myLed));  // toggle led
    } 
  } // if (!AHRS)
  else
  {
    myIMU.delt_t = millis() - myIMU.count;

    // update LCD once per half-second independent of read rate
    if (myIMU.delt_t > 14)
    {
      if(SerialDebug)
      {
        Serial.print("ax ");  Serial.print( aaFilteredF.x);
        Serial.print(" ay "); Serial.print( aaFilteredF.y);
        Serial.print(" az "); Serial.print( aaFilteredF.z);

        Serial.print(" gx "); Serial.print(myIMU.gx, 2);
        Serial.print(" gy "); Serial.print(myIMU.gy, 2);
        Serial.print(" gz "); Serial.print(myIMU.gz, 2);
        
        Serial.print(" mx "); Serial.print((int)myIMU.mx);
        Serial.print(" my "); Serial.print((int)myIMU.my);
        Serial.print(" mz "); Serial.print((int)myIMU.mz);
        
        Serial.print(" qw "); Serial.print(*getQ());
        Serial.print(" qx "); Serial.print(*(getQ() + 1));
        Serial.print(" qy "); Serial.print(*(getQ() + 2));
        Serial.print(" qz "); Serial.println(*(getQ() + 3));
      
      }

      myIMU.yaw   = atan2(2.0f * (*(getQ()+1) * *(getQ()+2) + *getQ()
                    * *(getQ()+3)), *getQ() * *getQ() + *(getQ()+1)
                    * *(getQ()+1) - *(getQ()+2) * *(getQ()+2) - *(getQ()+3)
                    * *(getQ()+3));
      myIMU.pitch = -asin(2.0f * (*(getQ()+1) * *(getQ()+3) - *getQ()
                    * *(getQ()+2)));
      myIMU.roll  = atan2(2.0f * (*getQ() * *(getQ()+1) + *(getQ()+2)
                    * *(getQ()+3)), *getQ() * *getQ() - *(getQ()+1)
                    * *(getQ()+1) - *(getQ()+2) * *(getQ()+2) + *(getQ()+3)
                    * *(getQ()+3));
      myIMU.pitch *= RAD_TO_DEG;
      myIMU.yaw   *= RAD_TO_DEG;
      myIMU.yaw   -= 1.55;
      myIMU.roll  *= RAD_TO_DEG;

/*
      if(SerialDebug)
      {
        Serial.print(" Yaw,Pitch,Roll ");
        Serial.print(myIMU.yaw, 2);
        Serial.print(" ");
        Serial.print(myIMU.pitch, 2);
        Serial.print(" ");
        Serial.println(myIMU.roll, 2);
      }

     if(count >200){
      old_v = new_v;
      new_v.x = trapezoid(myIMU.ax,  old_v.x, myIMU.delt_t); // reading out new velocity
      new_v.y = trapezoid(myIMU.ay,  old_v.y, myIMU.delt_t);
      new_v.z = trapezoid(myIMU.az,  old_v.z, myIMU.delt_t);
      // filtering v
      old_vLP = new_vLP;    
      old_vfilt = new_vfilt;    
      new_vLP = lowpassfilter(&old_vLP, &new_v , &old_v, 0.94); // calculate new drift offset
      new_vfilt.x = new_v.x - new_vLP.x;// subtract drift from data
      new_vfilt.y = new_v.y - new_vLP.y;
      new_vfilt.z = new_v.z - new_vLP.z;
      
      // updating displacement  
      old_s = new_s;                    // storing old data
      new_s.x = translation(new_vfilt.x, old_s.x, myIMU.ax, myIMU.delt_t); // reading out new data
      new_s.y = translation(new_vfilt.y, old_s.y, myIMU.ay, myIMU.delt_t);
      new_s.z = translation(new_vfilt.z, old_s.z, myIMU.az, myIMU.delt_t);
      //filtering s
      old_sLP = new_sLP;                // storing old drift offset
      old_sfilt = new_sfilt;            // storing old filtered data
      new_sLP = lowpassfilter(&old_sLP, &new_s , &old_s, 0.94); // calculate new drift offset
      new_sfilt.x = new_s.x - new_sLP.x;// subtract drift from data
      new_sfilt.y = new_s.y - new_sLP.y;
      new_sfilt.z = new_s.z - new_sLP.z;
     }

      Serial.print(" v "); Serial.print(new_vfilt.x);
      Serial.print(" ");   Serial.print(new_vfilt.y);
      Serial.print(" ");   Serial.print(new_vfilt.z);
      Serial.print(" p "); Serial.print(new_sfilt.x);
      Serial.print(" ");   Serial.print(new_sfilt.y);
      Serial.print(" ");   Serial.println(new_sfilt.z);*/
      count = count +1;

      myIMU.count = millis();
      myIMU.sumCount = 0;
      myIMU.sum = 0;
    } 
  } // if (AHRS)
}


// ================================================================
// ===                      FUNCTIONS                           ===
// ================================================================

VectorFloat noiseFilter(VectorFloat *compareValue, VectorInt16 *value, VectorFloat *previousCorrect, float noiseParameterLowPass, VectorInt16 *noiseParameterSen, VectorInt16 *noiseParameter  ) {
  VectorFloat filteredValue;
  if (value->x < compareValue->x - noiseParameter->x | value->x > compareValue->x + noiseParameter->x) {
    filteredValue.x = value->x;
  } else {
    if (value->x < compareValue->x - noiseParameterSen->x | value->x > compareValue->x + noiseParameterSen->x) {
      filteredValue.x = (float)(previousCorrect->x + value->x) / 2;
    } else {
      filteredValue.x = ((float)((1.0 - noiseParameterLowPass) * previousCorrect->x + noiseParameterLowPass * value->x));
    }
  }

  if (value->y < compareValue->y - noiseParameter->y | value->y > compareValue->y + noiseParameter->y) {
    filteredValue.y = value->y;
  } else {
    if (value->y < compareValue->y - noiseParameterSen->y | value->y > compareValue->y + noiseParameterSen->y) {
      filteredValue.y = (float)(previousCorrect->y + value->y) / 2;
    } else {
      filteredValue.y = ((float)((1.0 - noiseParameterLowPass) * previousCorrect->y + noiseParameterLowPass * value->y));
    }
  }

  if (value->z < compareValue->z - noiseParameter->z | value->z > compareValue->z + noiseParameter->z) {
    filteredValue.z = value->z;
  } else {
    if (value->z < compareValue->z - noiseParameterSen->z | value->z > compareValue->z + noiseParameterSen->z) {
      filteredValue.z = (float)(previousCorrect->z + value->z) / 2;
    } else {
      filteredValue.z = ((float)((1.0 - noiseParameterLowPass) * previousCorrect->z + noiseParameterLowPass * value->z));
    }

  }
  return filteredValue;
}

float trapezoid(float data,  float old_out, float dT) {
  float out;
  out = old_out + dT * data *0.001;
  return out;
}

float translation(float data,  float old_out, float data_a, float dT) {
  float out;
  out = old_out + dT * data *0.001 + 0.5*data_a* dT*0.001* dT*0.001;
  return out;
}

VectorFloat lowpassfilter(VectorFloat *prevX, VectorFloat *newU, VectorFloat *prevU, float alpha) {
  VectorFloat newX;
  //newX.x = ((1 - alpha) * prevX->x + alpha * prevU->x);
  //newX.y = ((1 - alpha) * prevX->y + alpha * prevU->y);
  //newX.z = ((1 - alpha) * prevX->z + alpha * prevU->z);
  newX.x = (alpha * ( - prevU->x + newU->x) + alpha * prevX->x);
  newX.y = (alpha * ( - prevU->y + newU->y) + alpha * prevX->y);
  newX.z = (alpha * ( - prevU->z + newU->z) + alpha * prevX->z);
  
  return newX;
}

  
//#############################################################################
