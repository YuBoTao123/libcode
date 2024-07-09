#include "crash.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdlib.h>

struct node {
    int data;
    struct node *next;
};

struct node *create_node(int data);
struct node *create_list(int length);
void print_list(struct node *list, int length);

int main(void){
    initCrashHandle();

    int length1 = 7;
    struct node *list1 = create_list(length1);
    print_list(list1, length1);

    return 0;
}

struct node *create_node(int data){
    struct node *new_node = (struct node *)malloc(sizeof(struct node));
    assert(new_node != NULL);
    new_node->data = data;
    new_node->next = NULL;
    return new_node;
}

struct node *create_list(int length) {

    struct node *head = NULL;
    if (length > 0) {
        head = create_node(0);
        int i = 1;
        struct node *curr = head;
        while (i < length) {
            curr->next = create_node(i);
            curr = curr->next;
            i++;
        }
    }
    return head;
}

void print_list(struct node *list, int length){
    struct node *curr = list;
    int i = 0;
    while (i <= length) {
        printf("%d->", curr->data);
        curr = curr->next;
        i++;
    }
    printf("X\n");
}
