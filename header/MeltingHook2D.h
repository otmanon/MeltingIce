#include "PhysicsHook.h"
#include "Domain.h"



class MeltingHook2D : public PhysicsHook
{
private:
	bool explicitMelting = true;
	bool implicitMelting = false;

	bool renderFluid = true;
	bool renderSolid = true;
	bool renderNormals = false;
	bool renderTGrad = false;
	bool renderVp = false;
	bool renderGlobal = false;
	bool renderVertexVel = true;
	bool renderInterpolatedVel = true;

	bool stepping_mode = false;
	bool stepping_flag = false;
	bool melting_flag = true;
	bool diffusion_flag = true;
	bool retriangulate = false;

	float vis_scale = 0.10;
	float phi = 50; //flux applied at each edge
	float dt = 1e-5; //timestep

	Eigen::MatrixXd origV; //original mesh description. Remember it when resetting
	Eigen::MatrixXi origF;

	Eigen::MatrixXd allV; // includes Solid geometry and liquid geometry.
	Eigen::MatrixXi allF; 
	Eigen::VectorXd allT;

	Eigen::MatrixXd renderV; // Mesh description that will be rendered (ie either solid geometry, liquid geometry, or both
	Eigen::MatrixXi renderF;
	Eigen::VectorXd renderT;

	std::string modelFilepath = "data/2dModels/elephant.off";

	Domain initMesh;
	Domain liquid;

	Domain D;
public:
	MeltingHook2D() : PhysicsHook() {}

	/*
	Initializes simulation. 
	*/
	virtual void initSimulation();

	/*
	Divides the domain into solid/liquid/boundary
	Input:
		V: num_Verts x 3 matrix of vertex coordinates
		E: Edges representing boundary we want to perform inside/outside segmentation of
	*/
	void segmentDomain(Eigen::MatrixXd V, Eigen::MatrixXi E, bool init);

	/*
	Calculates winding number of each face barycenter.
	Input:
		V - vertices of geometry
		F - faces of geomtry
		interfaceEdges - the interface with respect to which we are calculating winding numbers
	Output:
		W - Winding number of each barycenter
	*/
	void barycenterWindingNumber(Eigen::VectorXd& W, Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXi& interfaceEdges);
	
	/*
	Returns list of faces taht are inside and outside the interface in question
	W: winding number of each face barycenter
	solidF -List of vertex index triplets representing solid faces
	liquidF -List of vertex index triplets representing liquid faces
	*/
	void insideOutsideFaces(Eigen::MatrixXi& solidF, Eigen::MatrixXi& liquidF, Eigen::VectorXd& W);

	/*
	Given solid mesh, extracts its boundary, wraps a bounding box around it, triangulates 
	entire domain in one sweep, while maintaining elephant boundary edges
	*/
	void triangulateDomain(Eigen::MatrixXd V, Eigen::MatrixXi F, Eigen::MatrixXi E);

	/*
	Updates domain. When we melt, small triangles near the interface can occur. Need to handle those by either deleting unnecessary vertices
	or coming up with new triangle configuration.
	*/
	void updateDomain();

	/*
	For all interface Edges, finds the two corresponding Face indices in the globalF that surrounds each edge
	Useful for calculating interface normals
	*/
	void MeltingHook2D::interfaceEdgesToFaceAdjacency(Eigen::MatrixXi& solidF, Eigen::VectorXd & W);

	void retriangulateDomain();

	/*
	For each vertex index in the interface, give it a new local index. Used so that our sparse solve isn't so large
	*/
	void MeltingHook2D::interfaceGlobalToLocalVertexMappings();

	/*
	ImGui render... not sure how this works
	*/
	virtual void drawGUI(igl::opengl::glfw::imgui::ImGuiMenu &menu);

	/*
	Converts 3D vertices (used in libigl) to 2D (used in triangle)
	*/
	void convert3DVerticesTo2D(Eigen::MatrixXd& v3D, Eigen::MatrixXd& v2D);

	/*
	converts 2D vertices to 3D 
	*/
	void convert2DVerticesTo3D(Eigen::MatrixXd& v2D, Eigen::MatrixXd& v3D);

	/*
	Simulates one timestep
	*/
	virtual bool simulateOneStep();


	/*

	*/
	virtual void updateRenderGeometry();


	virtual void renderRenderGeometry(igl::opengl::glfw::Viewer &viewer);

	/*
	Steps simulation using explicit melting by representing Surface as border of triangle mesh.
	*/
	void explicitStep(float dt, float latentHeat);

	/*
	Steps simulation using an implicit enthalpy based approach, solving enthalpy diffusion, then extracting zero level set as surface
	*/
	void implicitStep(float dt, float latentHeat);
};