// OpenGL CUDA implementation of Julia Set. Try 3 X.X
// By Ray Imber a.k.a rayman22201
// Based off the book "CUDA by example"

#include "book.h" //includes stdio for me
#include "gl_helper.h"
#include "stdlib.h"

#include "cuda.h"
#include "cuda_gl_interop.h"

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;

#define     DIM    512

//animation clock
static float juliaClock = 0;
static int coin = 0;
static float *dev_juliaClock;

//debug float array
static float debug[(DIM*DIM)];
static float* dev_debug;
static FILE *dbgOut;

GLuint  bufferObj;
cudaGraphicsResource *resource;

//Julia Set functions
struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y, float clk ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c((-0.8 + clk), (0.156 + clk));
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return -1;
    }

	return a.magnitude2();
}

// based on ripple code, but uses uchar4 which is the type of data
// graphic inter op uses. see screenshot - basic2.png
__global__ void kernel( uchar4 *ptr , float *clockVal, float *dbg ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // now calculate the value at that position
	int juliaValue = julia( x, y , *clockVal );
	int red = 0;
	int green = 0;
	int blue = 0;
	int alpha = 255;
	dbg[offset] = juliaValue; //debug
	if( juliaValue != -1 )
	{
		//scale the julia value to be more useful for a color space.
		juliaValue = juliaValue * 1000000000;
		if((int)juliaValue < 10) { blue = 255; }
		else if((int)juliaValue < 50) { green = 255; }
		else { red = 255; }
		//debugging
		//printf("point ( %d , %d ) JuliaValue = %d \n",x,y,((int)juliaValue));
	}

    // accessing uchar4 vs unsigned char*
    ptr[offset].x = red;
    ptr[offset].y = green;
    ptr[offset].z = blue;
    ptr[offset].w = alpha;
}

//quit when esc is pressed
static void key_func( unsigned char key, int x, int y ) {
    switch (key) {
        case 27:
            // clean up OpenGL and CUDA
            HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
            glDeleteBuffers( 1, &bufferObj );
			HANDLE_ERROR( cudaFree( dev_juliaClock ) );
			HANDLE_ERROR( cudaFree( dev_debug ) ); //debug
			fclose(dbgOut);
            exit(0);
    }
}

//actually redraw the buffer
static void draw_func( void ) {
    // we pass zero as the last parameter, because out bufferObj is now
    // the source, and the field switches from being a pointer to a
    // bitmap to now mean an offset into a bitmap object
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );
    glDrawPixels( DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
    glutSwapBuffers();
}

//animation callback
static void idle_func( void ) {
	//update the animation clock
	if( juliaClock > 0.5) { coin = 1; }
	if( juliaClock < -0.5) { coin = 0; }
	if( coin == 0 ) { juliaClock = juliaClock + 0.001; }
	if( coin == 1 ) { juliaClock = juliaClock - 0.001; }
	//move the clock value to the GPU -- just overwrites any old value
	HANDLE_ERROR( cudaMemcpy( dev_juliaClock, &juliaClock, sizeof(float), cudaMemcpyHostToDevice ) );
	
	// do work with the memory dst being on the GPU, gotten via mapping
    HANDLE_ERROR( cudaGraphicsMapResources( 1, &resource, NULL ) );
    uchar4* devPtr;
    size_t  size;
    HANDLE_ERROR( 
        cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
                                              &size, 
                                              resource) );

    dim3    grids(DIM/16,DIM/16);
    dim3    threads(16,16);
    kernel<<<grids,threads>>>( devPtr , dev_juliaClock, dev_debug );
    HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &resource, NULL ) );
	//----------------------------------------------------------------

	//debug
	HANDLE_ERROR( cudaMemcpy( &debug, dev_debug, (sizeof(float)*DIM*DIM), cudaMemcpyDeviceToHost ) );
	fprintf(dbgOut,"Clock: %f array_copied\n",juliaClock);
	int i, j = 0;
	for(i = 0; i < DIM; i++) {
		for(j = 0; j < DIM; j++) {
			int dbgOffset = i + j*DIM;
			if(debug[dbgOffset] != -1){
				fprintf(dbgOut,"(%d,%d) JuliaValue: %f\n",i,j,debug[dbgOffset]);
			}
		}
	}
	fprintf(dbgOut,"----------------------------------------------\n\n");
	//force redisplay
	glutPostRedisplay();
}


int main( int argc, char **argv ) {
    cudaDeviceProp  prop;
    int dev;

//---------------initialize GLUT and CUDA----------------------------------------------------
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );

    // tell CUDA which dev we will be using for graphic interop
    // from the programming guide:  Interoperability with OpenGL
    //     requires that the CUDA device be specified by
    //     cudaGLSetGLDevice() before any other runtime calls.

    HANDLE_ERROR( cudaGLSetGLDevice( dev ) );

    // these GLUT calls need to be made before the other OpenGL
    // calls, else we get a seg fault
    glutInit( &argc, argv );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( DIM, DIM );
    glutCreateWindow( "test OpenGL" );

    glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
    glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
    glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
    glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

    // the first three are standard OpenGL, the 4th is the CUDA registration 
    // of the bitmap these calls exist starting in OpenGL 1.5
    glGenBuffers( 1, &bufferObj );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
    glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4,
                  NULL, GL_DYNAMIC_DRAW_ARB );

    HANDLE_ERROR( 
        cudaGraphicsGLRegisterBuffer( &resource, 
                                      bufferObj, 
                                      cudaGraphicsMapFlagsNone ) );

	HANDLE_ERROR( cudaMalloc( (void**)&dev_juliaClock, sizeof(float) ) );
	//-------------------------------------------------------------------------------------------

	//debug
	HANDLE_ERROR( cudaMalloc( (void**)&dev_debug, (sizeof(float)*DIM*DIM) ) );
	HANDLE_ERROR( cudaMemset(dev_debug,0,(sizeof(float)*DIM*DIM)) );
	dbgOut = fopen("JuliaDebug.txt","a+");

    // set up GLUT and kick off main loop
    glutKeyboardFunc( key_func );
    glutDisplayFunc( draw_func );
	glutIdleFunc( idle_func );
    glutMainLoop();
}