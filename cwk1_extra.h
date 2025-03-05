//
// Extra routines for the coursework, including for display. You are free to inspect this file but do not
// make any changes, as it will be overwritten with a different version during assessment.
//

// There is an optional graphical display; please see instructions on how to build to use this.
#ifdef GLFW
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>
#endif

// Parse the command line arguments. Returns -1 if there was a problem.
int parseCommandLineArguments( int argc, char **argv, int *N, int *M, int *nIters, int *version )
{
    // Need exactly 4 numbers on the command line, which means 5 arguments (note the executable itself is the first argument argv[0]).
    if( argc != 5 )
    {
        printf( "Incorrect number of command line arguments. Call as:\n" );
        printf( "./cwk1 <N> <M> <num> <version>\n" );
        printf( "where:\n - <N> sets the box size to be NxN.\n" );
        printf( " - <M> is the number of cells to add at the start.\n" );
        printf( " - <num> is the number of iterations to perform.\n" );
        printf( " - <version> is 0 or 1 for the original rule set or the modified version.\n" );
        return -1;
    }

    // System size. Note the grid is assumed to be square, i.e., NxN.
    *N = atoi( argv[1] );
    if( *N<3 )
    {
        printf( "Unsuitable box size N=%d - should be at least 3.\n", *N );
        return -1;
    }

    // Number of cells to add.
    *M = atoi( argv[2] );
    if( *M<0 || *M>((*N-2)*(*N-2)) )
    {
        printf( "Unsuitable initial number of cells M=%d. Should be at least 1, and no more than (N-2)x(N-2) as the boundary cells must remain empty.\n", *M );
        return -1;
    }

    // Number of iterations to perform.
    *nIters = atoi( argv[3] );
    if( *nIters<0 )
    {
        printf( "Unsuitable number of iterations %d: cannot be negative.\n", *nIters );
        return -1;
    }

    // Rule version.
    *version = atoi( argv[4] );
    if( *version!=0 && *version!=1 )
    {
        printf( "Unsuitable version number %d. Should be 0 for the original rules and 1 for the modified version.\n", *version );
        return -1;
    }

    // All fine.
    return 0;
}

// Allocate memory for the grid and its copy. Returns -1 if allocation fails.
int allocateGridsMemory( int ***grid1, int ***grid2, int N )
{
    int i;

    // Allocate N pointers for the grid rows.
    *grid1 = (int**) malloc( N*sizeof(int*) );
    if( !(*grid1) )
    {
        printf( "Could not allocate memory for the %d by %d grid. Try a smaller value of N.\n", N, N );
        return -1;
    }
    
    // Allocate the actual rows and set to zero.
    for( i=0; i<N; i++ )
    {
        (*grid1)[i] = (int*) calloc( N, sizeof(int) );
        if( !(*grid1)[i] )
        {
            printf( "Could not allocate memory for the %d by %d grid. Try a smaller value of N.\n", N, N );
            return -1;
        }
    }

    // Same again for grid2.
    *grid2 = (int**) malloc( N*N*sizeof(int) );
    if( !(*grid2) )
    {
        printf( "Could not allocate memory for a copy of the grid.\n" );
        return -1;
    }

    for( i=0; i<N; i++ )
    {
        (*grid2)[i] = (int*) malloc( N*sizeof(int) );
        if( !(*grid2)[i] )
        {
            printf( "Could not allocate memory for the %d by %d grid. Try a smaller value of N.\n", N, N );
            return -1;
        }
    }

    return 0;
}

// Textual display of the grid to the terminal.
void displayGrid( int **grid, int N, int numCells )
{
    int i, j;

    for( i=N-1; i>=0; i-- )       // Reverse order so top corresponds to highest row.
    {
        for( j=0; j<N; j++ )
            printf( grid[i][j] ? "O" : "." );       // 'O' for a cell, '.' for an empty grid square.

        printf( "\n" );     // Newline at the end of each grid row.
    }
    printf( " - %d cells.\n", numCells );
}


//
// The remaining functions in this file are only used if graphical output (OpenGL/GLFW) are selected and available.
//
#ifdef GLFW

// This routine actually displays the image.
void displayImage( int N, int **grid )
{
	// Clear the window, in white.
    glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );
	glClear( GL_COLOR_BUFFER_BIT );

    // Cells always drawn in black.
    glColor3f( 0.0f, 0.0f, 0.0f );

    // Size of a grid square in the default OpenGL window, which varies from -1 to 1 in each axis.
    float dx = 2.0f / N;

    int i, j;
	for( j=0; j<N; j++ )
		for( i=0; i<N; i++ )
            if( grid[j][i] )
            {
                glBegin( GL_POLYGON );
                glVertex3f( -1.0f +  i   *dx, -1.0f +  j   *dx, 0.0f );
                glVertex3f( -1.0f + (i+1)*dx, -1.0f +  j   *dx, 0.0f );
                glVertex3f( -1.0f + (i+1)*dx, -1.0f + (j+1)*dx, 0.0f );
                glVertex3f( -1.0f +  i   *dx, -1.0f + (j+1)*dx, 0.0f );
                glEnd();
		    }
}

// Error callback function for GLFW.
void graphicsErrorCallBack( int errorCode, const char* message )
{
    printf( "Error with OpenGL/GLFW: code %i, message %s\n", errorCode, message );
}

// Keyboard callback: Checks for escape or 'Q'uit.
void keyboardCallBack( GLFWwindow* window, int key, int scancode, int action, int mods )
{
    // Close if the escape key or 'q' is pressed.
    if( (key==GLFW_KEY_ESCAPE||key==GLFW_KEY_Q) && action==GLFW_PRESS )
        glfwSetWindowShouldClose( window, 1 );
}

// Initialise the GLFW window. Returns with the window handle.
GLFWwindow* initialiseGLFWWindow( int N, int **grid )
{
    // Set grid square size to give a reasonable sized window. May fail for very large systems!
    int squareSize = 600 / N;
    if( squareSize<1 ) squareSize=1;
    int windowSize_x = squareSize*N, windowSize_y = squareSize*N;

    GLFWwindow* window;

    if( !glfwInit() ) return NULL;

    window = glfwCreateWindow( windowSize_x, windowSize_y, "Once the simulation has finished, press 'q' or <ESC> to close this window", NULL, NULL );
    if( !window )
    {
        glfwTerminate();
        return NULL;
    }

    glfwSetErrorCallback( graphicsErrorCallBack );
    glfwSetKeyCallback( window, keyboardCallBack );
    glfwMakeContextCurrent(window);

    return window;
}

#endif
