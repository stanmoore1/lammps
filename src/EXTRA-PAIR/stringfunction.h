#ifndef STRINGFUNCTION__
#define STRINGFUNCTION__

typedef double (*unary_func) (double x);
typedef double (*binary_func) (double x,double y);
typedef void (*two_in_two_out) (double x,double y,double dx[1],double dy[1]);

typedef double (*funt)(double x);
typedef struct NODE {
  int type,narg;
  double val,dval;
  funt func,dfunc;
  int eval_tag;
  struct NODE *args[2];
} node;

enum { CONSTANT,PARAMETER,ONEFUNCTION,TWOFUNCTION,COMMUTATIVE,LINEAR_TREE,NTYPES };


void delete_tree(node *tree);
double eval_tree(node *tree,int nparms,double parms[]) ;
double eval_tree_deriv(node *tree,int nparms,double parms[][2]) ;

node *parse_string(char *s,int nparms,char *parms[]);

#endif
