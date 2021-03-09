// BiTree.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include"malloc.h"

typedef struct BiTNode{
	char data;
	struct BiTNode *lchild, *rchild;
}BiTNode,*BiTree;

//访问的具体操作
void visit(char i, int level) {
	printf("%c位于%d层",i, level);
}

//遍历二叉树
void PreOrderTraverse(BiTree T, int level) {
	if (T) {
		visit(T->data, level);
		PreOrderTraverse(T->lchild, level + 1);
		PreOrderTraverse(T->rchild, level + 1);
	}
}


//创建一颗二叉树 前序遍历
void CreateBitree(BiTree *T) {
	char c;
	scanf("%c", &c);
	if (' ' == c) {
		*T = NULL;
	}
	else {
		*T = (BiTNode *)malloc(sizeof(BiTNode));
		(*T)->data = c;
		CreateBitree(&(*T)->lchild);
		CreateBitree(&(*T)->rchild);
	}
}

int main() {
	int level = 1;
	BiTree T = NULL;
	CreateBitree(&T);
	printf("2");
	PreOrderTraverse(T, level);
	printf("3");
	return 0;
}