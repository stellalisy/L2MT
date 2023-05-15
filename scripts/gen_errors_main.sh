
for i in {16..101}; do
    # qdel $((3642890 + i))
    qsub /home/sli136/l2mt/scripts/gen_errors_sub.sh para ${i}
done

# for i in {5..101}; do
#     # qdel $((3642692 + i))
#     qsub /home/sli136/l2mt/scripts/gen_errors_sub.sh para ${i}
# done

# 1 to 60 already submitted