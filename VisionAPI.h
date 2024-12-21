#pragma once


#ifdef CREATEDLL_EXPORTS 
#define VISIONAI_DECLSPEC __declspec(dllexport)
#else
#define VISIONAI_DECLSPEC __declspec(dllimport)
#endif

//extern "C" VISIONAI_DECLSPEC void Init();

/**
*
* 
*/